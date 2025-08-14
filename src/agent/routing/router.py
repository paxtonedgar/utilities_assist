# src/agent/routing/router.py
"""
Routing logic for LangGraph workflows.

Extracted from graph.py to follow Single Responsibility Principle.
Contains all routing decisions and conditions.
"""

from typing import Literal, Dict, Any
import logging

# Import constants
from agent.constants import NORMALIZED_QUERY

def to_state_dict(state):
    """
    Convert any state object (GraphState, Pydantic BaseModel, or dict) to a plain dict.
    
    Handles:
    - Plain dict: return as-is
    - Pydantic v2: use model_dump()
    - Pydantic v1: use dict()
    - Any other object: attempt dict() conversion
    """
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):       # pydantic v2
        return state.model_dump()
    if hasattr(state, "dict"):             # pydantic v1
        return state.dict()
    return dict(state)  # last resort

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
    def route_after_intent(cls, state) -> RouteDestination:
        """Route to appropriate search based on intent classification."""
        from agent.nodes.base_node import get_intent_label
        
        # Convert GraphState to dict for safe access
        s = to_state_dict(state)
        intent = s.get("intent")
        if not intent:
            logger.warning("No intent found in state, defaulting to confluence search")
            return "search_confluence"
        
        # Use safe intent label extraction to prevent AttributeError
        intent_label = get_intent_label(intent)
        if not intent_label:
            logger.warning("No intent label found, defaulting to confluence search")
            return "search_confluence"
        
        intent_type = intent_label.lower()
        
        # Handle special intents with direct routing
        if intent_type in cls.INTENT_ROUTES:
            route = cls.INTENT_ROUTES[intent_type]
            logger.info(f"Routing to {route} based on intent: {intent_type}")
            return route
        
        # Check for complex queries needing multi-search
        if cls._needs_multi_search(s, intent_type):
            logger.info(f"Routing to multi-search for complex query: {intent_type}")
            return "search_multi"
        
        # Default to confluence search
        logger.info(f"Routing to confluence search for intent: {intent_type}")
        return "search_confluence"
    
    @classmethod
    def route_after_rewrite(cls, state) -> RewriteRouteDestination:
        """Route after query rewrite - use same logic as initial routing."""
        # Reuse the same logic but filter to valid rewrite destinations
        # Note: route_after_intent already handles state normalization
        initial_route = cls.route_after_intent(state)
        
        # Map to valid rewrite destinations
        rewrite_routes = {
            "search_confluence": "search_confluence",
            "search_swagger": "search_swagger", 
            "search_multi": "search_multi"
        }
        
        return rewrite_routes.get(initial_route, "search_confluence")
    
    @classmethod
    def _needs_multi_search(cls, state_dict: Dict[str, Any], intent_type: str) -> bool:
        """Determine if query needs multi-search based on complexity indicators."""
        query = state_dict.get("normalized_query", "").lower()
        
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
        state, 
        coverage_threshold: float = 0.7, 
        min_results: int = 3
    ) -> CoverageRouteDestination:
        """
        Check if search results meet coverage requirements.
        
        Args:
            state: Current graph state (GraphState or dict)
            coverage_threshold: Minimum coverage score required
            min_results: Minimum number of results required
            
        Returns:
            "rewrite" if coverage is insufficient, "combine" if acceptable
        """
        # Convert GraphState to dict for safe access
        s = to_state_dict(state)
        search_results = s.get("search_results", [])
        loop_count = s.get("loop_count", 0) 
        rewrite_attempts = s.get("rewrite_attempts", 0)
        normalized_query = s.get(NORMALIZED_QUERY, "")
        
        # Hard stop on AttributeError exceptions (intent object crashes)
        error_messages = s.get("error_messages", [])
        if error_messages:
            for error in error_messages:
                if any(keyword in error for keyword in ["AttributeError", "IntentResult", "'intent'", "has no attribute"]):
                    logger.warning(f"Stopping rewrite: prior step threw AttributeError on intent - {error[:100]}...")
                    return "combine"
        
        # Prevent infinite loops with multiple checks
        if loop_count >= 3:
            logger.warning(f"Max loops reached ({loop_count}), proceeding to combine")
            return "combine"
        
        # Short-circuit on too many rewrite attempts
        if rewrite_attempts >= 2:
            logger.warning(f"Max rewrite attempts reached ({rewrite_attempts}), proceeding to combine")
            return "combine"
        
        # If query is empty or too short, proceed to combine (don't rewrite)
        if not normalized_query or len(normalized_query.strip()) < 3:
            logger.warning("Query is empty or too short, proceeding to combine")
            return "combine"
        
        # Check result count
        if len(search_results) < min_results:
            logger.info(f"Insufficient results ({len(search_results)} < {min_results}), rewriting query")
            return "rewrite"
        
        # Check coverage score if available
        # This would need to be implemented based on your coverage calculation
        coverage_score = cls._calculate_coverage_score(s)
        if coverage_score < coverage_threshold:
            logger.info(f"Low coverage score ({coverage_score:.2f} < {coverage_threshold}), rewriting query")
            return "rewrite"
        
        logger.info(f"Coverage acceptable ({coverage_score:.2f}), proceeding to combine")
        return "combine"
    
    @classmethod
    def _calculate_coverage_score(cls, state_dict: Dict[str, Any]) -> float:
        """
        Calculate coverage score based on search results.
        
        This is a placeholder - implement based on your coverage logic.
        """
        search_results = state_dict.get("search_results", [])
        if not search_results:
            return 0.0
        
        # Simple coverage based on average score
        avg_score = sum(r.score for r in search_results) / len(search_results)
        return avg_score