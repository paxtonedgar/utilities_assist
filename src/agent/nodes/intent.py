"""Intent node - intent classification using regex-only slotter."""

import logging
from pydantic import BaseModel, Field

from src.services.models import IntentResult
from src.telemetry.logger import stage
from .base_node import to_state_dict, from_state_dict
from src.agent.intent.slotter import slot, SlotResult

# Import constants to prevent KeyError issues
from src.agent.constants import NORMALIZED_QUERY, INTENT

logger = logging.getLogger(__name__)


def _map_slot_to_legacy_intent(slot_result: SlotResult) -> str:
    """Map new slotter intent to legacy intent format for backward compatibility."""
    # Analyze reasons and features to determine best legacy mapping
    reasons = slot_result.reasons

    # Check for API/swagger indicators
    if any(
        "api" in reason.lower() or "endpoint" in reason.lower() for reason in reasons
    ):
        return "swagger"

    # Check for list/catalog indicators
    if any(
        "list" in reason.lower() or "available" in reason.lower() for reason in reasons
    ):
        return "list"

    # Check for procedure indicators
    if slot_result.intent == "procedure" and slot_result.doish:
        return "workflow"

    # Default mapping based on slotter intent
    intent_mapping = {
        "info": "confluence",
        "procedure": "workflow",
        "mixed": "confluence",  # Default to confluence for mixed intent
    }

    return intent_mapping.get(slot_result.intent, "confluence")


class IntentAnalysis(BaseModel):
    """Structured output for intent classification."""

    intent: str = Field(description="Primary intent category")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the classification")


@stage("classify_intent")
async def intent_node(state, config, *, store=None):
    """
    Classify query intent using regex-only slotter (eliminates ALL LLM calls).

    Follows LangGraph pattern with config parameter and optional store injection.

    Args:
        state: Workflow state (GraphState or dict) containing normalized_query
        config: RunnableConfig with user context and configuration
        store: Optional BaseStore for cross-thread user memory

    Returns:
        State update with intent classification (same type as input)
    """
    incoming_type = type(state)
    s = to_state_dict(state)

    # Get query for analysis
    normalized_query = s.get(NORMALIZED_QUERY, "")

    if not normalized_query or normalized_query.strip() == "":
        logger.warning(
            "intent_node: normalized_query missing/empty; using default confluence intent"
        )
        default_intent = IntentResult(
            intent="confluence",
            confidence=0.5,
            reasoning="Default intent for empty query",
        )
        return from_state_dict(incoming_type, {**s, INTENT: default_intent})

    # PERFORMANCE: Use regex-only slotter for ALL queries (eliminates LLM calls)
    try:
        slot_result = slot(normalized_query)
        logger.info(
            f"Regex slotter classification: {slot_result.intent} (confidence: {slot_result.confidence:.2f}, reasons: {slot_result.reasons}, saved ~4s LLM call)"
        )

        # Map slotter intent to legacy intent format
        legacy_intent = _map_slot_to_legacy_intent(slot_result)

        intent_result = IntentResult(
            intent=legacy_intent,
            confidence=slot_result.confidence,
            reasoning=f"Slotter: {', '.join(slot_result.reasons)}",
        )
        return from_state_dict(incoming_type, {**s, INTENT: intent_result})

    except Exception as e:
        logger.error(f"Regex slotter failed: {e}")
        # Hard fallback to default intent (no LLM)
        default_intent = IntentResult(
            intent="confluence",
            confidence=0.5,
            reasoning=f"Default fallback after slotter failure: {e}",
        )
        return from_state_dict(incoming_type, {**s, INTENT: default_intent})


# Wrapper for LangGraph tool registration
async def wrapped_intent_node(state, config=None):
    """Wrapper to handle the case where config might be None in some LangGraph versions."""
    if config is None:
        config = {"configurable": {}}

    try:
        return await intent_node(state, config)
    except Exception as e:
        logger.error(f"Intent node failed: {e}")
        # Return state with default intent on complete failure
        s = to_state_dict(state)
        default_intent = IntentResult(
            intent="confluence", confidence=0.5, reasoning="Error fallback"
        )
        return from_state_dict(type(state), {**s, INTENT: default_intent})


# Keep the original function signature for compatibility
intent_tool = wrapped_intent_node
