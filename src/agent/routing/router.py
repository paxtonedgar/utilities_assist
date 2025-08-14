# src/agent/routing/router.py
"""
Routing logic for LangGraph workflows.

Extracted from graph.py to follow Single Responsibility Principle.
Contains all routing decisions and conditions.
"""

from typing import Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Type alias for route destinations
RouteDestination = Literal[
    "search_confluence", "search_swagger", "search_multi", 
    "list_handler", "workflow_synthesizer", "restart"
]

RewriteRouteDestination = Literal["search_confluence", "search_swagger", "search_multi"]

CoverageRouteDestination = Literal["rewrite", "combine"]


class IntentRouter:
    """Routes based on intent classification with clear decision logic."""
    
    # Mapping of intents to routes
    INTENT_ROUTES = {
        "restart": "restart",
        "list": "list_handler", 
        "workflow": "workflow_synthesizer",
        "swagger": "search_swagger"
    }
    
    # Keywords that indicate comparative queries needing multi-search
    COMPARATIVE_KEYWORDS = ["compare", "versus", "vs", "difference", "which is better"]
    MULTI_ENTITY_INDICATORS = ["and", "both", "either", "all"]
    
    @classmethod
    def route_after_intent(cls, state: Dict[str, Any]) -> RouteDestination:
        """Route to appropriate search based on intent classification."""
        intent = state.get("intent")
        if not intent:
            logger.warning("No intent found in state, defaulting to confluence search")
            return "search_confluence"
        
        intent_type = intent.intent.lower()
        
        # Handle special intents with direct routing
        if intent_type in cls.INTENT_ROUTES:
            route = cls.INTENT_ROUTES[intent_type]
            logger.info(f"Routing to {route} based on intent: {intent_type}")
            return route
        
        # Check for complex queries needing multi-search
        if cls._needs_multi_search(state, intent_type):
            logger.info(f"Routing to multi-search for complex query: {intent_type}")
            return "search_multi"
        
        # Default to confluence search
        logger.info(f"Routing to confluence search for intent: {intent_type}")
        return "search_confluence"
    
    @classmethod
    def route_after_rewrite(cls, state: Dict[str, Any]) -> RewriteRouteDestination:
        """Route after query rewrite - use same logic as initial routing."""
        # Reuse the same logic but filter to valid rewrite destinations
        initial_route = cls.route_after_intent(state)
        
        # Map to valid rewrite destinations
        rewrite_routes = {
            "search_confluence": "search_confluence",
            "search_swagger": "search_swagger", 
            "search_multi": "search_multi"
        }
        
        return rewrite_routes.get(initial_route, "search_confluence")
    
    @classmethod
    def _needs_multi_search(cls, state: Dict[str, Any], intent_type: str) -> bool:
        """Determine if query needs multi-search based on complexity indicators."""
        query = state.get("normalized_query", "").lower()
        
        # Check for comparative language
        is_comparative = any(keyword in query for keyword in cls.COMPARATIVE_KEYWORDS)
        
        # Check for multiple entities
        has_multiple_entities = any(indicator in query for indicator in cls.MULTI_ENTITY_INDICATORS)
        
        # Check intent type
        is_complex_intent = intent_type in ["workflow", "comparative"]
        
        return is_comparative or has_multiple_entities or is_complex_intent


class CoverageChecker:
    """Checks search coverage and determines if rewriting is needed."""
    
    @classmethod
    def check_coverage(
        cls, 
        state: Dict[str, Any], 
        coverage_threshold: float = 0.7, 
        min_results: int = 3
    ) -> CoverageRouteDestination:
        """
        Check if search results meet coverage requirements.
        
        Args:
            state: Current graph state
            coverage_threshold: Minimum coverage score required
            min_results: Minimum number of results required
            
        Returns:
            "rewrite" if coverage is insufficient, "combine" if acceptable
        """
        search_results = state.get("search_results", [])
        loop_count = state.get("loop_count", 0)
        
        # Prevent infinite loops
        if loop_count >= 3:
            logger.warning(f"Max loops reached ({loop_count}), proceeding to combine")
            return "combine"
        
        # Check result count
        if len(search_results) < min_results:
            logger.info(f"Insufficient results ({len(search_results)} < {min_results}), rewriting query")
            return "rewrite"
        
        # Check coverage score if available
        # This would need to be implemented based on your coverage calculation
        coverage_score = cls._calculate_coverage_score(state)
        if coverage_score < coverage_threshold:
            logger.info(f"Low coverage score ({coverage_score:.2f} < {coverage_threshold}), rewriting query")
            return "rewrite"
        
        logger.info(f"Coverage acceptable ({coverage_score:.2f}), proceeding to combine")
        return "combine"
    
    @classmethod
    def _calculate_coverage_score(cls, state: Dict[str, Any]) -> float:
        """
        Calculate coverage score based on search results.
        
        This is a placeholder - implement based on your coverage logic.
        """
        search_results = state.get("search_results", [])
        if not search_results:
            return 0.0
        
        # Simple coverage based on average score
        avg_score = sum(r.score for r in search_results) / len(search_results)
        return avg_score