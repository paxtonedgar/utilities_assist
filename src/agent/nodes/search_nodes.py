# src/agent/nodes/search_nodes.py
"""Search Node for the opinionated Jarvis path.

Linear pipeline: summarize -> plan -> search(per-aspect) -> compose -> answer
"""

import logging
from typing import Dict, Any, List
from .base_node import BaseNodeHandler
from src.agent.constants import SEARCH_QUERY
from src.agent.tools.search import (
    search_api_docs,
    search_procedures,
    search_general,
)
from src.infra.resource_manager import get_resources
from src.services.models import Passage

logger = logging.getLogger(__name__)


"""IntentNode removed: planning now drives routing and structure."""


class SearchNode(BaseNodeHandler):
    """Search node using evidence-based approach."""
    
    def __init__(self):
        super().__init__("search")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute search logic."""
        try:
            resources = get_resources()
            if not resources:
                logger.error("Resources not available for search")
                return self._handle_search_error(state, "Resources unavailable")
            
            query = state.get(SEARCH_QUERY, state.get("query_normalized", state.get("normalized_query", "")))
            plan = state.get("plan")
            filters = build_filters_from_state(state)
            
            if not query:
                logger.warning("No query available for search")
                return self._handle_search_error(state, "No query provided")
            
            # Plan-driven per-aspect search (no branching, small k per aspect)
            if isinstance(plan, dict) and plan.get("aspects"):
                aspects = plan.get("aspects", [])
                plan_filters = plan.get("filters") or {}
                # Merge filters (state-derived < plan-derived)
                merged_filters = {**(filters or {}), **plan_filters}
                k_per_aspect = int(plan.get("k_per_aspect", 3))
                strategies = plan.get("aspect_strategies") or {}

                sections: Dict[str, List[Passage]] = {}
                flat: List[Passage] = []
                # Prepare query variants: acronym form + expanded form
                q_variants = [query]
                try:
                    from src.agent.acronym_map import expand_acronym
                    exp_q, expansions = expand_acronym(query)
                    if expansions:
                        # Include the bare expansion phrase and the expanded query text
                        q_variants.append(expansions[0])
                        q_variants.append(exp_q)
                except Exception:
                    pass

                for aspect in aspects:
                    strategy = strategies.get(
                        aspect,
                        "bm25" if aspect in ("steps", "api", "troubleshoot") else "enhanced_rrf",
                    )
                    if aspect == "api":
                        results_lists = []
                        for q in q_variants:
                            r = await search_api_docs(
                                q, resources.search_client, resources.embed_client,
                                resources.settings.embed.model, filters=merged_filters, strategy=strategy
                            )
                            results_lists.append(r.passages)
                        keep = _dedupe_chain(results_lists, k_per_aspect)
                    elif aspect == "steps":
                        results_lists = []
                        for q in q_variants:
                            r = await search_procedures(
                                q, resources.search_client, resources.embed_client,
                                resources.settings.embed.model, filters=merged_filters, strategy=strategy
                            )
                            results_lists.append(r.passages)
                        keep = _dedupe_chain(results_lists, k_per_aspect)
                    else:
                        results_lists = []
                        for q in q_variants:
                            r = await search_general(
                                q, resources.search_client, resources.embed_client,
                                resources.settings.embed.model, filters=merged_filters, strategy=strategy
                            )
                            results_lists.append(r.passages)
                        keep = _dedupe_chain(results_lists, k_per_aspect)
                    sections[aspect] = keep
                    flat.extend(keep)

                if not flat:
                    return self._handle_empty_results(state, query)

                return {
                    **state,
                    "search_sections": self._to_serializable_sections(sections),
                    "search_results": flat,
                    "total_results": len(flat),
                    "search_strategy": "plan_per_aspect",
                    "workflow_path": state.get("workflow_path", []) + ["search"],
                }
            else:
                logger.error("Plan missing or empty; refusing legacy routing")
                return self._handle_search_error(state, "plan_missing")

            # No legacy comparison path; planner would specify a compare aspect in the future
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._handle_search_error(state, str(e))
    
    async def _handle_comparison_search(self, state: Dict[str, Any], query: str, resources) -> Dict[str, Any]:
        """Handle comparison search separately."""
        comparison_result = await search_comparison(
            query, resources.search_client, resources.embed_client, 
            resources.settings.embed.model
        )
        
        if not comparison_result:
            # Fall back to general search
            logger.info("Comparison extraction failed, falling back to general search")
            search_result = await search_general(
                query, resources.search_client, resources.embed_client,
                resources.settings.embed.model
            )
            
            return {
                **state,
                "search_results": search_result.passages,
                "search_strategy": "comparison_fallback_to_general",
                "workflow_path": state.get("workflow_path", []) + ["search_comparison"]
            }
        
        # Create comparison briefing directly
        comparison_briefing = render_comparison_briefing(comparison_result, query)
        
        return {
            **state,
            "search_results": comparison_result.merged_results,
            "comparison_result": {
                "entity1": comparison_result.entity1,
                "entity2": comparison_result.entity2,
                "confidence": comparison_result.comparison_confidence
            },
            "final_briefing": comparison_briefing,
            "briefing_ready": True,  # Skip evidence composer for comparisons
            "workflow_path": state.get("workflow_path", []) + ["search_comparison"]
        }
    
    async def _execute_specialized_search(self, route_action: str, query: str, resources, filters: Optional[Dict[str, Any]]):
        """Execute appropriate search based on route."""
        if route_action == "api":
            return await search_api_docs(
                query, resources.search_client, resources.embed_client,
                resources.settings.embed.model, filters=filters
            )
        elif route_action == "procedure": 
            return await search_procedures(
                query, resources.search_client, resources.embed_client,
                resources.settings.embed.model, filters=filters
            )
        elif route_action == "list":
            # For list queries, use general search but with specific content filtering
            options = SearchOptions(
                content_boost={ContentType.LIST_DATA: 2.0},
                top_k=25  # Get more results for lists
            )
            return await search_docs(
                query, resources.search_client, resources.embed_client,
                resources.settings.embed.model, options, filters=filters
            )
        else:  # general and default
            return await search_general(
                query, resources.search_client, resources.embed_client,
                resources.settings.embed.model, filters=filters
            )
    
    def _handle_search_error(self, state: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Handle search errors gracefully."""
        return {
            **state,
            "search_results": [],
            "search_error": error_msg,
            "workflow_path": state.get("workflow_path", []) + ["search_error"]
        }
    
    def _handle_empty_results(self, state: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle empty search results."""
        no_results_msg = (
            f"No relevant information found for: {query}. "
            "Try rephrasing your question or being more specific."
        )
        
        return {
            **state,
            "search_results": [],
            "final_briefing": no_results_msg,
            "briefing_ready": True,  # Skip evidence composer
            "workflow_path": state.get("workflow_path", []) + ["search_empty"]
        }

    def _to_serializable_sections(self, sections: Dict[str, List[Passage]]):
        """Convert Passage objects to plain dicts for composer service."""
        out: Dict[str, List[Dict]] = {}
        for aspect, items in sections.items():
            rows: List[Dict] = []
            for p in items:
                meta = getattr(p, "meta", {}) or {}
                rows.append(
                    {
                        "doc_id": p.doc_id,
                        "title": getattr(p, "title", None) or meta.get("title"),
                        "url": getattr(p, "url", None),
                        "snippet": getattr(p, "text", ""),
                        "score": float(getattr(p, "score", 0.0)),
                        "heading": meta.get("heading"),
                    }
                )
            out[aspect] = rows
        return out


def _dedupe_chain(lists_of_passages: List[List[Passage]], k: int) -> List[Passage]:
    """Merge multiple passage lists, dedupe by doc_id, keep first-seen/top-scored order.

    This allows us to search both acronym and expansion variants and keep the
    best small set per aspect.
    """
    merged: List[Passage] = []
    seen = set()
    for plist in lists_of_passages:
        for p in plist:
            if p.doc_id not in seen:
                merged.append(p)
                seen.add(p.doc_id)
                if len(merged) >= k:
                    return merged
    return merged[:k]


def build_filters_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ACL/space/time filters from state for downstream search."""
    filters: Dict[str, Any] = {}
    for key in ("acl_hash", "space_key", "content_type"):
        if state.get(key):
            filters[key] = state[key]
    for key in ("updated_after", "updated_before"):
        if state.get(key):
            filters[key] = state[key]
    user_filters = state.get("user_filters")
    if isinstance(user_filters, dict):
        filters.update(user_filters)
    return filters


class CombineNode(BaseNodeHandler):
    """Evidence-gated briefing composer node."""
    
    def __init__(self):
        super().__init__("combine")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute evidence-based briefing composition."""
        # Skip if briefing already ready (e.g., from comparison search)
        if state.get("briefing_ready", False):
            logger.info("Briefing already prepared, skipping evidence composer")
            return {
                **state,
                "workflow_path": state.get("workflow_path", []) + ["combine_skipped"]
            }
        
        search_results = state.get("search_results", [])
        query = state.get(SEARCH_QUERY, state.get("query_normalized", state.get("normalized_query", "")))
        
        if not search_results:
            logger.warning("No search results available for evidence composition")
            return {
                **state,
                "final_briefing": "No information found to compose a briefing.",
                "workflow_path": state.get("workflow_path", []) + ["combine_empty"]
            }
        
        try:
            # Compose evidence-gated briefing
            logger.info(f"🔍 EVIDENCE: Using query for evidence composition: '{query}'")
            composition = compose_evidence_briefing(search_results, query, max_sections=4)
            
            if not composition.sections:
                # Fallback to simple presentation
                fallback_briefing = self._create_fallback_briefing(search_results, query)
                logger.info("No evidence sections found, using fallback briefing")
                
                return {
                    **state,
                    "final_briefing": fallback_briefing,
                    "composition_strategy": "fallback_simple",
                    "workflow_path": state.get("workflow_path", []) + ["combine_fallback"]
                }
            
            # Render structured briefing
            briefing_markdown = render_briefing_markdown(composition)
            
            logger.info(f"Evidence briefing composed: {len(composition.sections)} sections, strategy: {composition.composition_strategy}")
            logger.info(f"📝 FINAL_BRIEFING: Setting final_briefing with {len(briefing_markdown) if briefing_markdown else 0} characters")
            
            return {
                **state,
                "final_briefing": briefing_markdown,
                "composition": {
                    "sections": list(composition.sections.keys()),
                    "strategy": composition.composition_strategy,
                    "total_passages": composition.total_passages
                },
                "workflow_path": state.get("workflow_path", []) + ["combine"]
            }
            
        except Exception as e:
            logger.error(f"Evidence composition failed: {e}")
            fallback_briefing = self._create_fallback_briefing(search_results, query)
            
            return {
                **state,
                "final_briefing": fallback_briefing,
                "composition_error": str(e),
                "workflow_path": state.get("workflow_path", []) + ["combine_error"]
            }
    
    def _create_fallback_briefing(self, results: List[Passage], query: str) -> str:
        """Create simple fallback briefing when evidence composition fails."""
        if not results:
            return f"No information found for: {query}"
        
        lines = [
            f"# {query}",
            "",
            f"Found {len(results)} relevant sources:",
            "",
        ]
        
        for i, result in enumerate(results[:5], 1):
            title = result.meta.get("title", f"Source {i}")
            score = result.score
            snippet = result.text[:200] + "..." if len(result.text) > 200 else result.text
            
            lines.extend([
                f"## {i}. {title}",
                f"*Relevance: {score:.2f}*",
                "",
                snippet,
                "",
            ])
        
        return "\n".join(lines)


# Specialized search node variants
class ConfluenceSearchNode(SearchNode):
    """Confluence-specific search node."""
    def __init__(self):
        super().__init__()
        self.node_name = "search_confluence"


class SwaggerSearchNode(SearchNode):
    """Swagger/API documentation search node."""
    def __init__(self):
        super().__init__()
        self.node_name = "search_swagger"


class MultiSearchNode(SearchNode):
    """Multi-index search node."""
    def __init__(self):
        super().__init__()
        self.node_name = "search_multi"
