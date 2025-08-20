# src/quality/utils.py
"""
Centralized utilities for coverage evaluation system.
Eliminates code duplication across router.py and search_to_compose.py
"""

from typing import List, Dict, Any
from .coverage import CoverageGate, Passage
from .subquery import decompose

# Single global coverage gate instance (eliminating duplication)
_COVERAGE_GATE = None

def get_coverage_gate() -> CoverageGate:
    """Get singleton coverage gate instance with configuration."""
    global _COVERAGE_GATE
    if _COVERAGE_GATE is None:
        try:
            from src.infra.settings import get_settings
            settings = get_settings()
            
            # Use reranker config for model and thresholds if available
            model_name = "BAAI/bge-reranker-v2-m3"  # Default to BGE
            device = "auto"
            tau = 0.25  # Updated threshold for BGE scores
            
            if hasattr(settings, 'reranker') and settings.reranker and settings.reranker.enabled:
                # Use BGE v2-m3 reranker configuration
                model_name = settings.reranker.model_id
                device = settings.reranker.device
                tau = settings.reranker.min_score  # Use reranker threshold
                
            # Use coverage config if available, otherwise use defaults
            if hasattr(settings, 'coverage') and settings.coverage:
                coverage_config = settings.coverage
                weights = {
                    "steps": coverage_config.weight_steps,
                    "endpoint": coverage_config.weight_endpoint,
                    "jira": coverage_config.weight_jira,
                    "owner": coverage_config.weight_owner,
                    "table": coverage_config.weight_table,
                }
                _COVERAGE_GATE = CoverageGate(
                    model_name=model_name,
                    device=device,
                    tau=tau,  # Use BGE-appropriate threshold
                    alpha=coverage_config.alpha,
                    gate_ar=coverage_config.gate_ar,
                    gate_andcg=coverage_config.gate_andcg,
                    min_actionable_spans=coverage_config.min_actionable_spans,
                    weights=weights
                )
            else:
                # Use defaults with BGE model
                _COVERAGE_GATE = CoverageGate(
                    model_name=model_name,
                    device=device,
                    tau=tau,  # BGE threshold
                    alpha=0.5,
                    gate_ar=0.60, gate_andcg=0.40,
                    min_actionable_spans=3
                )
        except Exception as e:
            # Fallback to BGE defaults on any config error
            _COVERAGE_GATE = CoverageGate(
                model_name="BAAI/bge-reranker-v2-m3",
                device="auto",
                tau=0.25,
                alpha=0.5,
                gate_ar=0.60, gate_andcg=0.40,
                min_actionable_spans=3
            )
    return _COVERAGE_GATE


def convert_search_results_to_passages(search_results: List, for_coverage: bool = True) -> List[Dict[str, Any]]:
    """
    Convert SearchResult objects to passage dictionaries.
    Centralizes the conversion logic used across multiple files.
    
    Args:
        search_results: List of SearchResult objects
        for_coverage: Whether to format for coverage evaluation (includes rank, etc.)
        
    Returns:
        List of passage dictionaries with keys: text, url, title, heading, rank
    """
    passages = []
    for idx, result in enumerate(search_results):
        passage = {
            "text": getattr(result, 'content', '') or getattr(result, 'text', ''),
            "url": getattr(result, 'url', ''),
            "title": getattr(result, 'title', ''),
            "heading": result.metadata.get('heading', '') if hasattr(result, 'metadata') else '',
        }
        
        if for_coverage:
            passage["rank"] = idx + 1
            
        passages.append(passage)
    
    return passages


def create_passage_objects(passage_dicts: List[Dict[str, Any]]) -> List[Passage]:
    """
    Convert passage dictionaries to Passage objects.
    Centralizes the Passage object creation logic.
    
    Args:
        passage_dicts: List of dictionaries with passage data
        
    Returns:
        List of Passage objects
    """
    return [Passage(text=p["text"], meta={k: p.get(k) for k in ("url","title","heading","rank")})
            for p in passage_dicts]


def run_coverage_evaluation(user_query: str, search_results: List) -> Dict[str, Any]:
    """
    Complete coverage evaluation pipeline.
    Centralizes the full evaluation process used across multiple files.
    
    Args:
        user_query: User's query string
        search_results: List of SearchResult objects
        
    Returns:
        Dict with coverage evaluation results
    """
    if not search_results or not user_query:
        return {
            "gate_pass": False,
            "aspect_recall": 0.0,
            "alpha_ndcg": 0.0,
            "actionable_spans": 0,
            "subqueries": [],
            "selected_count": 0
        }
    
    # Convert to passage format
    passage_dicts = convert_search_results_to_passages(search_results, for_coverage=True)
    passages = create_passage_objects(passage_dicts)
    
    # Generate subqueries
    subqs = decompose(user_query)
    
    # Evaluate coverage
    coverage_gate = get_coverage_gate()
    ev = coverage_gate.evaluate(user_query, subqs, passages)
    
    # Select passages
    picks = sorted({j for idxs in ev["picks"].values() for j in idxs})
    selected = [passage_dicts[j] for j in picks if j < len(passage_dicts)]
    
    return {
        "gate_pass": ev["gate_pass"],
        "aspect_recall": ev["aspect_recall"],
        "alpha_ndcg": ev["alpha_ndcg"],
        "actionable_spans": ev["actionable_spans"],
        "subqueries": subqs,
        "selected_count": len(selected),
        "selected_passages": selected,
        "all_passages": passage_dicts
    }