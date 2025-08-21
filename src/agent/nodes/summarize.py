"""Summarize node - query normalization using simple rules (no LLM)."""

import logging

from src.services.normalize import normalize_query
from src.telemetry.logger import stage
from .base_node import BaseNodeHandler, to_state_dict, from_state_dict

# Import constants to prevent KeyError issues
from src.agent.constants import ORIGINAL_QUERY, NORMALIZED_QUERY

logger = logging.getLogger(__name__)


@stage("normalize")
async def summarize_node(state, config, *, store=None):
    """
    Summarize/normalize the user query using simple rules (eliminates ALL LLM calls).

    Follows LangGraph pattern with config parameter and optional store injection.

    Args:
        state: Workflow state (GraphState or dict) containing original_query
        config: RunnableConfig with user context and configuration
        store: Optional BaseStore for cross-thread user memory

    Returns:
        State update with normalized_query (same type as input)
    """
    import time
    from src.infra.telemetry import log_normalize_stage

    start_time = time.perf_counter()
    req_id = getattr(config, "run_id", "unknown") if config else "unknown"
    incoming_type = type(state)

    # Convert to dict for processing
    s = to_state_dict(state)

    # STATE_CHECK: Verify query preservation before processing
    logger.info(
        "NODE_START summarize | keys=%s | original=%r | normalized=%r",
        list(s.keys()),
        s.get(ORIGINAL_QUERY),
        s.get(NORMALIZED_QUERY),
    )

    # Use consistent state keys with fallback
    user_input = s.get(ORIGINAL_QUERY) or s.get(NORMALIZED_QUERY, "")
    if not user_input:
        logger.warning(
            "summarize_node: empty input; keys=%s original=%r normalized=%r",
            list(s.keys()),
            s.get(ORIGINAL_QUERY, None),
            s.get(NORMALIZED_QUERY, None),
        )
        # CRITICAL: Preserve ALL existing state fields and return proper type
        merged = {
            **s,  # Preserve all existing state
            NORMALIZED_QUERY: "",
            "workflow_path": s.get("workflow_path", []) + ["summarize_empty"],
        }
        return from_state_dict(incoming_type, merged)

    # PERFORMANCE: Use simple rule-based normalization (eliminates LLM calls)
    try:
        normalized = normalize_query(user_input)
        logger.info(
            f"Rule-based normalized: '{user_input}' -> '{normalized}' (saved ~2s LLM call)"
        )
    except Exception as e:
        logger.error(f"Rule-based normalization failed: {e}")
        # Ultimate fallback: use original query
        normalized = user_input
        logger.info(f"Using original query as normalized: '{normalized}'")

    # Add telemetry logging
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    log_normalize_stage(req_id, user_input, normalized, elapsed_ms)

    # STATE_CHECK: Verify result before return
    merged = {
        **s,  # Preserve all existing state
        NORMALIZED_QUERY: normalized,
        "workflow_path": s.get("workflow_path", []) + ["summarize"],
    }

    logger.info(
        "NODE_END summarize | normalized=%r | workflow_path=%s",
        normalized,
        merged.get("workflow_path", []),
    )

    # CRITICAL: Preserve ALL existing state fields and return proper type
    return from_state_dict(incoming_type, merged)


class SummarizeNode(BaseNodeHandler):
    """Class-based wrapper for summarize functionality."""

    def __init__(self):
        super().__init__("summarize")

    async def execute(self, state: dict, config: dict = None) -> dict:
        """Execute the summarize logic using the existing function."""
        # The existing function expects config parameter
        if config is None:
            config = {"configurable": {"thread_id": "unknown"}}
        return await summarize_node(state, config)
