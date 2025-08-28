"""Summarize node - query normalization using simple rules (no LLM)."""

import logging
import re
import json
from functools import lru_cache
from typing import Dict

from src.telemetry.logger import stage
from .base_node import BaseNodeHandler, to_state_dict, from_state_dict

# Import constants to prevent KeyError issues
from src.agent.constants import ORIGINAL_QUERY, NORMALIZED_QUERY

logger = logging.getLogger(__name__)


# === MIGRATED FROM SERVICES.NORMALIZE ===

# Optional synonym caching - can be disabled if synonyms change frequently
@lru_cache(maxsize=1)  # Single synonym dict only
def _load_synonyms() -> Dict[str, str]:
    """Load and cache synonym mappings."""
    try:
        # Try to load synonyms from data directory
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        synonyms_path = os.path.join(base_dir, "data", "synonyms.json")

        if os.path.exists(synonyms_path):
            with open(synonyms_path, "r", encoding="utf-8") as f:
                synonyms = json.load(f)
                logger.info(f"Loaded {len(synonyms)} synonym mappings")
                return synonyms
        else:
            logger.warning(f"Synonyms file not found at {synonyms_path}")
    except Exception as e:
        logger.error(f"Failed to load synonyms: {e}")

    # Fallback to basic synonyms
    return {
        "csu": "Customer Summary Utility",
        "customer summary": "Customer Summary Utility",
        "gcp": "Global Customer Platform",
        "etu": "Enhanced Transaction Utility",
        "etu-au": "Enhanced Transaction Utility Account Utility",
        "etu au": "Enhanced Transaction Utility Account Utility",
        "transaction utility": "Enhanced Transaction Utility",
        "au": "Account Utility",
        "ciu": "Customer Interaction Utility",
        "customer interaction utility": "Customer Interaction Utility",
        "de": "Digital Events",
        "pcu": "Product Catalog Utility",
        "digev": "Digital Events",
        "apg": "APG",
    }


def normalize_query(text: str) -> str:
    """Normalize user input by replacing synonyms and cleaning text."""
    if not text or not text.strip():
        return ""

    # Get synonym mappings
    synonym_mapping = _load_synonyms()

    # Convert all synonym keys to lowercase for case-insensitive matching
    lower_synonyms = {key.lower(): value for key, value in synonym_mapping.items()}

    # Build regex pattern for all synonyms (case-insensitive)
    if not lower_synonyms:
        return text.strip()

    pattern = (
        r"\b("
        + "|".join(re.escape(synonym) for synonym in lower_synonyms.keys())
        + r")\b"
    )

    def replace_match(match):
        """Replace matched synonym with canonical form."""
        return lower_synonyms[match.group(0).lower()]

    # Apply synonym replacement
    normalized_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

    # Additional normalization
    normalized_text = normalized_text.strip()
    normalized_text = re.sub(r"\s+", " ", normalized_text)  # Normalize whitespace

    if normalized_text != text.strip():
        logger.info(f"Normalized '{text.strip()}' -> '{normalized_text}'")

    return normalized_text


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
