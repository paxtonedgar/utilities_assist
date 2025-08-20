# src/agent/nodes/search_to_compose.py
"""
Search result processing with coverage gate integration.
Replaces the old handwritten coverage math with academic IR-style assessment.
"""

import logging
import time
from typing import Dict, Any, List
from src.quality.coverage import CoverageGate, Passage
from src.quality.subquery import decompose
from src.services.models import SearchResult
from src.infra.telemetry import log_coverage_stage
from .base_node import to_state_dict, from_state_dict

logger = logging.getLogger(__name__)

# Global coverage gate instance (initialized lazily)
_COVERAGE_GATE = None

def get_coverage_gate():
    """Get or create the coverage gate instance with configuration."""
    global _COVERAGE_GATE
    if _COVERAGE_GATE is None:
        try:
            from src.infra.settings import get_settings
            settings = get_settings()
            
            # Use coverage config if available, otherwise use defaults
            if settings.coverage:
                coverage_config = settings.coverage
                weights = {
                    "steps": coverage_config.weight_steps,
                    "endpoint": coverage_config.weight_endpoint,
                    "jira": coverage_config.weight_jira,
                    "owner": coverage_config.weight_owner,
                    "table": coverage_config.weight_table,
                }
                _COVERAGE_GATE = CoverageGate(
                    model_name=coverage_config.model_name,
                    tau=coverage_config.tau,
                    alpha=coverage_config.alpha,
                    gate_ar=coverage_config.gate_ar,
                    gate_andcg=coverage_config.gate_andcg,
                    min_actionable_spans=coverage_config.min_actionable_spans,
                    weights=weights
                )
            else:
                # Use defaults
                _COVERAGE_GATE = CoverageGate(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    tau=0.45, alpha=0.5,
                    gate_ar=0.60, gate_andcg=0.40,
                    min_actionable_spans=3
                )
            logger.info("Initialized coverage gate with configuration")
        except Exception as e:
            logger.warning(f"Failed to load coverage config, using defaults: {e}")
            _COVERAGE_GATE = CoverageGate(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                tau=0.45, alpha=0.5,
                gate_ar=0.60, gate_andcg=0.40,
                min_actionable_spans=3
            )
    return _COVERAGE_GATE

async def search_coverage_node(state, config=None, *, store=None):
    """
    Apply coverage gate to search results before combining/answering.
    
    Replaces the old coverage math with cross-encoder answerability scoring
    and aspect recall + alpha-nDCG evaluation.
    
    Args:
        state: Workflow state containing search_results
        config: RunnableConfig with user context
        store: Optional BaseStore for cross-thread memory
        
    Returns:
        State update with coverage evaluation and filtered results
    """
    incoming_type = type(state)
    
    try:
        # Convert to dict for processing
        s = to_state_dict(state)
        
        logger.info(
            "NODE_START search_coverage | keys=%s | search_results_count=%d",
            list(s.keys()),
            len(s.get("search_results", []))
        )
        
        user_query = s.get("normalized_query", s.get("original_query", ""))
        search_results = s.get("search_results", [])
        req_id = s.get("request_id", "unknown")
        
        if not user_query.strip():
            logger.warning("Empty query for coverage evaluation")
            log_coverage_stage(req_id, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, "Empty query")
            return from_state_dict(incoming_type, {
                **s,
                "coverage_evaluation": {"gate_pass": False, "error": "Empty query"},
                "workflow_path": s.get("workflow_path", []) + ["coverage_empty_query"]
            })
        
        if not search_results:
            logger.info("No search results for coverage evaluation")
            log_coverage_stage(req_id, False, 0.0, 0.0, 0, 0, 0, 0, 0.0, "No search results")
            return from_state_dict(incoming_type, {
                **s,
                "coverage_evaluation": {"gate_pass": False, "error": "No search results"},
                "workflow_path": s.get("workflow_path", []) + ["coverage_no_results"]
            })
        
        # Run coverage evaluation with timing
        start_time = time.perf_counter()
        result = run_search_and_gate(user_query, search_results)
        ms = (time.perf_counter() - start_time) * 1000
        
        # Log telemetry for successful evaluation
        log_coverage_stage(
            req_id,
            result["gate_pass"],
            result["metrics"]["aspect_recall"],
            result["metrics"]["alpha_ndcg"],
            result["metrics"]["actionable_spans"],
            result["metrics"]["selected_count"],
            len(search_results),
            len(result["metrics"]["subqueries"]),
            ms
        )
        
        logger.info(
            "NODE_END search_coverage | gate_pass=%s | aspect_recall=%.3f | alpha_ndcg=%.3f | selected=%d",
            result["gate_pass"],
            result["metrics"]["aspect_recall"],
            result["metrics"]["alpha_ndcg"],
            result["metrics"]["selected_count"]
        )
        
        # Update state with coverage results
        merged = {
            **s,
            "coverage_evaluation": result["metrics"],
            "gate_pass": result["gate_pass"],
            "selected_passages": result["selected_passages"],
            "all_passages_debug": result.get("all_passages_debug", search_results),
            "workflow_path": s.get("workflow_path", []) + ["coverage_gate"],
            "performance_metrics": {
                **s.get("performance_metrics", {}),
                "coverage_aspect_recall": result["metrics"]["aspect_recall"],
                "coverage_alpha_ndcg": result["metrics"]["alpha_ndcg"],
                "coverage_actionable_spans": result["metrics"]["actionable_spans"]
            }
        }
        
        return from_state_dict(incoming_type, merged)
        
    except Exception as e:
        logger.error(f"Coverage evaluation failed: {e}")
        s = to_state_dict(state) if 's' not in locals() else s
        req_id = s.get("request_id", "unknown")
        
        # Log telemetry for error case
        log_coverage_stage(req_id, False, 0.0, 0.0, 0, 0, len(s.get("search_results", [])), 0, 0.0, str(e))
        
        # Fallback - mark as failed gate
        merged = {
            **s,
            "coverage_evaluation": {"gate_pass": False, "error": str(e)},
            "gate_pass": False,
            "workflow_path": s.get("workflow_path", []) + ["coverage_error"],
            "error_messages": s.get("error_messages", []) + [f"Coverage evaluation failed: {e}"]
        }
        return from_state_dict(incoming_type, merged)


def run_search_and_gate(user_query: str, search_results: List[SearchResult]) -> dict:
    """
    Apply coverage gate to search results.
    
    Args:
        user_query: Original user query
        search_results: List of SearchResult objects from search
        
    Returns:
        Dict with gate_pass, metrics, selected_passages, etc.
    """
    # 1) Sub-queries
    subqs = decompose(user_query)
    
    # 2) Convert SearchResult objects to Passage objects
    passages = []
    for idx, result in enumerate(search_results):
        # Extract metadata from SearchResult
        meta = {
            "url": getattr(result, 'url', ''),
            "title": getattr(result, 'title', ''),
            "heading": result.metadata.get('heading', ''),
            "rank": idx + 1  # Use position in list as rank
        }
        
        # Get text content
        text = getattr(result, 'content', '') or getattr(result, 'text', '')
        
        passages.append(Passage(text=text, meta=meta))
    
    # 3) Evaluate coverage
    coverage_gate = get_coverage_gate()
    ev = coverage_gate.evaluate(user_query, subqs, passages)
    
    # 4) Choose passages to send to composer: best 1 per aspect
    picks = sorted({j for idxs in ev["picks"].values() for j in idxs})
    selected = [search_results[j] for j in picks if j < len(search_results)]
    
    # If no passages were selected but we have results, select top 3 as fallback
    if not selected and search_results:
        selected = search_results[:3]
        logger.info("No passages met coverage threshold, using top 3 as fallback")
    
    return {
        "gate_pass": ev["gate_pass"],
        "metrics": {
            "aspect_recall": round(ev["aspect_recall"], 3),
            "alpha_ndcg": round(ev["alpha_ndcg"], 3),
            "actionable_spans": int(ev["actionable_spans"]),
            "subqueries": subqs,
            "selected_count": len(selected),
        },
        "selected_passages": selected,  # feed only these to the composer
        "all_passages_debug": search_results,  # keep for logs if needed
    }


def convert_search_results_to_passages(search_results: List[SearchResult]) -> List[dict]:
    """
    Convert SearchResult objects to simple dicts for coverage evaluation.
    
    Args:
        search_results: List of SearchResult objects
        
    Returns:
        List of dict with keys: text, url, title, heading, rank
    """
    passages = []
    for idx, result in enumerate(search_results):
        passages.append({
            "text": getattr(result, 'content', '') or getattr(result, 'text', ''),
            "url": getattr(result, 'url', ''),
            "title": getattr(result, 'title', ''),
            "heading": result.metadata.get('heading', ''),
            "rank": idx + 1
        })
    return passages