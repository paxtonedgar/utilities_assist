# src/agent/routing/router.py
"""
Routing logic for LangGraph workflows.

Extracted from graph.py to follow Single Responsibility Principle.
Contains all routing decisions and conditions.
"""

from typing import Literal, Dict, Any
import logging

# Import constants and utilities
from src.agent.constants import NORMALIZED_QUERY
from src.agent.nodes.base_node import to_state_dict

# Coverage evaluation now handled upstream in search tool before cross-encoder


logger = logging.getLogger(__name__)

# Type alias for route destinations
RouteDestination = Literal[
    "search_confluence",
    "search_swagger",
    "search_multi",
    "list_handler",
    "workflow_synthesizer",
    "restart",
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
        "swagger": "search_swagger",
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

        # ACRONYM GUARDRAIL: Pin domain to Utilities for short acronym queries
        query = s.get(NORMALIZED_QUERY, s.get("original_query", ""))
        if cls._is_utilities_acronym_query(query):
            logger.info(f"Acronym guardrail: pinning '{query}' to Utilities domain")
            # Force utilities search regardless of intent
            return "search_confluence"

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
            "search_multi": "search_multi",
        }

        return rewrite_routes.get(initial_route, "search_confluence")

    @classmethod
    def _is_utilities_acronym_query(cls, query: str) -> bool:
        """
        Check if query is a short acronym that should be pinned to Utilities domain.

        Uses loaded synonyms.json to detect known utility acronyms.
        Short (≤3 tokens) + matches acronym key → pin to Utilities.
        """
        if not query or len(query.strip()) == 0:
            return False

        tokens = query.strip().split()

        # Must be short query (≤3 tokens)
        if len(tokens) > 3:
            return False

        # Check if any token is a known utilities acronym
        try:
            from agent.acronym_map import _load_acronym_data

            utilities_acronyms = _load_acronym_data()

            for token in tokens:
                if token.upper() in utilities_acronyms:
                    logger.info(
                        f"Found utilities acronym: {token.upper()} → {utilities_acronyms[token.upper()]}"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Could not load acronym data for guardrail: {e}")

        return False

    @classmethod
    def _needs_multi_search(cls, state_dict: Dict[str, Any], intent_type: str) -> bool:
        """Determine if query needs multi-search based on complexity indicators."""
        query = state_dict.get("normalized_query", "").lower()

        # Check for comparative language
        is_comparative = any(keyword in query for keyword in cls.COMPARATIVE_KEYWORDS)

        # Check for multiple entities
        has_multiple_entities = any(
            indicator in query for indicator in cls.MULTI_ENTITY_INDICATORS
        )

        # Check intent type
        is_complex_intent = intent_type in ["workflow", "comparative"]

        return is_comparative or has_multiple_entities or is_complex_intent


class CoverageChecker:
    """Checks search coverage and determines if rewriting is needed."""

    @classmethod
    def check_coverage(cls, state, min_results: int = 3) -> CoverageRouteDestination:
        """
        Check if search results meet coverage requirements using cross-encoder gate.

        Args:
            state: Current graph state (GraphState or dict)
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
                if any(
                    keyword in error
                    for keyword in [
                        "AttributeError",
                        "IntentResult",
                        "'intent'",
                        "has no attribute",
                    ]
                ):
                    logger.warning(
                        f"Stopping rewrite: prior step threw AttributeError on intent - {error[:100]}..."
                    )
                    return "combine"

        # Prevent infinite loops with multiple checks
        if loop_count >= 3:
            logger.warning(f"Max loops reached ({loop_count}), proceeding to combine")
            return "combine"

        # Short-circuit on too many rewrite attempts
        if rewrite_attempts >= 2:
            logger.warning(
                f"Max rewrite attempts reached ({rewrite_attempts}), proceeding to combine"
            )
            return "combine"

        # If query is empty or too short, proceed to combine (don't rewrite)
        if not normalized_query or len(normalized_query.strip()) < 3:
            logger.warning("Query is empty or too short, proceeding to combine")
            return "combine"

        # CRITICAL: Never rewrite utility acronym queries - they are specific and targeted
        if IntentRouter._is_utilities_acronym_query(normalized_query):
            logger.info(
                f"Utility acronym query detected: '{normalized_query}' - skipping rewrite to preserve domain context"
            )
            return "combine"

        # PERFORMANCE FIX: Coverage gating now happens BEFORE cross-encoder in search tool
        # This eliminates the expensive rerank→coverage fail→rewrite loop
        # If we get here, coverage was already validated or cross-encoder was skipped

        # Simple result count check (much cheaper than full coverage evaluation)
        if len(search_results) < min_results:
            logger.info(
                f"Insufficient results ({len(search_results)} < {min_results}), rewriting query"
            )
            return "rewrite"

        # Coverage already validated upstream - proceed to combine
        logger.info(
            f"Coverage validation completed upstream, proceeding to combine with {len(search_results)} results"
        )
        return "combine"

    # REMOVED: Old coverage evaluation methods
    # Coverage gating now happens upstream in search tool before expensive cross-encoder
    # This eliminates the rerank→coverage fail→rewrite loop that was causing 7-8s delays
