# src/agent/nodes/search_nodes.py
"""
Search Node Handlers: Simplified nodes using evidence-based architecture.

Linear pipeline: micro_route -> search -> evidence_composer -> briefing
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base_node import BaseNodeHandler
from src.agent.routing.micro_router import micro_route, RouteResult
from src.agent.constants import SEARCH_QUERY
from src.agent.tools.search import search_docs, search_api_docs, search_procedures, search_general, SearchOptions, ContentType
from src.search.comparison_search import search_comparison, render_comparison_briefing
from src.agent.nodes.combine import compose_evidence_briefing, render_briefing_markdown
from src.infra.resource_manager import get_resources
from src.services.models import Passage

logger = logging.getLogger(__name__)


class IntentNode(BaseNodeHandler):
    """Intent/routing node using simple regex patterns."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute micro-routing logic."""
        query = state.get(SEARCH_QUERY, state.get("normalized_query", state.get("original_query", "")))
        
        if not query or not query.strip():
            logger.warning("Empty query provided to router")
            return {
                **state,
                "route_result": None,
                "next_action": "general",
                "workflow_path": state.get("workflow_path", []) + ["router_empty"]
            }
        
        # Use micro-router for simple, transparent routing
        route_result = micro_route(query)
        
        logger.info(f"Micro-route decision: '{query}' -> {route_result.route} ({route_result.confidence:.1f})")
        
        return {
            **state,
            "route_result": asdict(route_result),
            "next_action": route_result.route,
            "query_normalized": route_result.query_normalized,
            "workflow_path": state.get("workflow_path", []) + ["intent"]
        }


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
            route_action = state.get("next_action", "general")
            filters = self._build_filters_from_state(state)
            
            if not query:
                logger.warning("No query available for search")
                return self._handle_search_error(state, "No query provided")
            
            # Handle comparison queries specially
            if route_action == "compare":
                return await self._handle_comparison_search(state, query, resources)
            
            # Handle other route types with specialized search functions
            search_result = await self._execute_specialized_search(route_action, query, resources, filters)
            
            if not search_result or not search_result.passages:
                logger.warning(f"No results found for query: '{query}'")
                return self._handle_empty_results(state, query)
            
            logger.info(f"Search completed: {len(search_result.passages)} results for '{query}'")
            
            return {
                **state,
                "search_results": search_result.passages,
                "total_results": search_result.total_found,
                "search_strategy": search_result.search_strategy,
                "workflow_path": state.get("workflow_path", []) + ["search"]
            }
            
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

    def _build_filters_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
