# src/util/filters.py
"""
Consistent filter management to prevent filter state flipping mid-search loop.
Maintains filter consistency across rewrite passes and view switching.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterState:
    """Immutable filter state for a search session."""

    base_filters: Optional[Dict[str, Any]]
    intent_type: str
    view_type: str  # "info" or "procedure"
    enabled: bool

    def to_opensearch_filters(self) -> Optional[Dict[str, Any]]:
        """Convert to OpenSearch filter format."""
        if not self.enabled or not self.base_filters:
            return None
        return self.base_filters


class FilterManager:
    """Manages filter consistency across search passes."""

    def __init__(self):
        self._session_filters: Dict[str, FilterState] = {}

    def get_or_create_filters(
        self,
        req_id: str,
        intent_type: str,
        view_type: str,
        force_enabled: Optional[bool] = None,
    ) -> FilterState:
        """
        Get consistent filters for a request, creating if needed.

        Args:
            req_id: Request ID for consistency tracking
            intent_type: Intent classification (confluence, swagger, etc)
            view_type: View type (info, procedure)
            force_enabled: Override filter enablement

        Returns:
            FilterState with consistent configuration
        """
        if req_id in self._session_filters:
            existing = self._session_filters[req_id]
            logger.debug(
                f"Reusing existing filters for req_id={req_id}: enabled={existing.enabled}"
            )
            return existing

        # Create new filter state
        base_filters = self._build_base_filters(intent_type, view_type)
        enabled = force_enabled if force_enabled is not None else bool(base_filters)

        filter_state = FilterState(
            base_filters=base_filters,
            intent_type=intent_type,
            view_type=view_type,
            enabled=enabled,
        )

        # Cache for consistency
        self._session_filters[req_id] = filter_state

        logger.info(
            f"Created consistent filters for req_id={req_id}: intent={intent_type}, view={view_type}, enabled={enabled}"
        )

        return filter_state

    def _build_base_filters(
        self, intent_type: str, view_type: str
    ) -> Optional[Dict[str, Any]]:
        """Build base filters based on intent and view type."""

        # Intent-based routing
        if intent_type == "swagger" or intent_type == "api":
            return {
                "bool": {
                    "should": [
                        {"term": {"content_type": "swagger"}},
                        {"term": {"content_type": "api"}},
                        {"match": {"title": "api"}},
                        {"match": {"text": "endpoint"}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        # View-based optimization
        if view_type == "procedure":
            return {
                "bool": {
                    "should": [
                        {"term": {"content_type": "runbook"}},
                        {"term": {"content_type": "guide"}},
                        {"term": {"content_type": "tutorial"}},
                        {"term": {"content_type": "onboarding"}},
                        {"match": {"title": "how to"}},
                        {"match": {"title": "setup"}},
                        {"match": {"title": "onboarding"}},
                        {"match": {"text": "step"}},
                        {"match": {"text": "procedure"}},
                    ],
                    "minimum_should_match": 0,  # Boost, don't require
                }
            }

        elif view_type == "info":
            return {
                "bool": {
                    "should": [
                        {"term": {"content_type": "confluence"}},
                        {"term": {"content_type": "documentation"}},
                        {"term": {"content_type": "overview"}},
                        {"match": {"title": "overview"}},
                        {"match": {"title": "definition"}},
                        {"match": {"text": "what is"}},
                    ],
                    "minimum_should_match": 0,  # Boost, don't require
                }
            }

        # Default: no specific filtering (broader search)
        return None

    def clear_session(self, req_id: str) -> None:
        """Clear cached filters for a session."""
        self._session_filters.pop(req_id, None)
        logger.debug(f"Cleared filter session for req_id={req_id}")

    def get_filter_summary(self, req_id: str) -> str:
        """Get summary of current filter state for logging."""
        if req_id not in self._session_filters:
            return "no_filters"

        state = self._session_filters[req_id]
        return f"{state.intent_type}_{state.view_type}_{'enabled' if state.enabled else 'disabled'}"


# Global filter manager instance
_filter_manager = FilterManager()


def get_consistent_filters(
    req_id: str,
    intent_type: str = "confluence",
    view_type: str = "info",
    force_enabled: Optional[bool] = None,
) -> FilterState:
    """
    Get consistent filters for a search request.

    This function ensures filters don't flip state during rewrite loops,
    maintaining search consistency across passes.

    Args:
        req_id: Request ID for tracking consistency
        intent_type: Intent classification result
        view_type: Search view type (info/procedure)
        force_enabled: Force filter enablement override

    Returns:
        FilterState with consistent configuration

    Example:
        # First call - creates filters
        filters = get_consistent_filters("abc123", "confluence", "procedure")

        # Subsequent calls - returns same filter state
        filters = get_consistent_filters("abc123", "api", "info")  # Still procedure filters!
    """
    return _filter_manager.get_or_create_filters(
        req_id, intent_type, view_type, force_enabled
    )


def clear_filter_session(req_id: str) -> None:
    """Clear filter session when request completes."""
    _filter_manager.clear_session(req_id)


def get_filter_summary(req_id: str) -> str:
    """Get filter summary for telemetry."""
    return _filter_manager.get_filter_summary(req_id)
