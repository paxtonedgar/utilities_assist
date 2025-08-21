# src/agent/nodes/base_node.py
"""
Base node handler for LangGraph nodes with DRY principles.

Eliminates wrapper function repetition by providing common error handling,
logging, and state management patterns.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Type

from src.telemetry.logger import log_event

logger = logging.getLogger(__name__)


def to_state_dict(state: Union[Dict, Any]) -> Dict[str, Any]:
    """
    Convert any state object (GraphState, Pydantic BaseModel, or dict) to a plain dict.

    This prevents "GraphState object has no attribute 'get'" runtime errors
    by ensuring consistent dict access patterns throughout the LangGraph pipeline.

    Handles:
    - Plain dict: return as-is
    - Pydantic v2: use model_dump()
    - Pydantic v1: use dict()
    - GraphState instances: convert to dict
    - Any other object: attempt dict() conversion
    """
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):  # pydantic v2
        logger.debug("Converting Pydantic v2 model to dict")
        return state.model_dump()
    if hasattr(state, "dict"):  # pydantic v1
        logger.debug("Converting Pydantic v1 model to dict")
        return state.dict()

    # GraphState instances or other iterables
    try:
        result = dict(state)
        logger.debug(
            f"Converted {type(state).__name__} to dict with {len(result)} keys"
        )
        return result
    except Exception as e:
        logger.warning(f"Failed to convert {type(state).__name__} to dict: {e}")
        return {}  # Return empty dict as safe fallback


def get_intent_label(intent) -> str:
    """
    Safely extract intent label from either IntentResult object or dict.

    This prevents AttributeError crashes when intent format is inconsistent.

    Args:
        intent: Either IntentResult object, dict, or None

    Returns:
        Intent label string or None if not found
    """
    if intent is None:
        return None

    # Handle IntentResult object (has .intent attribute)
    if hasattr(intent, "intent"):
        return getattr(intent, "intent", None)

    # Handle dict form
    if isinstance(intent, dict):
        return intent.get("intent")

    # Fallback for string or other types
    if isinstance(intent, str):
        return intent

    logger.warning(f"Unknown intent type: {type(intent)} - {intent}")
    return None


def get_intent_confidence(intent) -> float:
    """
    Safely extract intent confidence from either IntentResult object or dict.

    Args:
        intent: Either IntentResult object, dict, or None

    Returns:
        Intent confidence float or 0.0 if not found
    """
    if intent is None:
        return 0.0

    # Handle IntentResult object (has .confidence attribute)
    if hasattr(intent, "confidence"):
        return getattr(intent, "confidence", 0.0)

    # Handle dict form
    if isinstance(intent, dict):
        return intent.get("confidence", 0.0)

    logger.warning(f"Unknown intent type for confidence: {type(intent)} - {intent}")
    return 0.0


def from_state_dict(state_type: Type, data: Dict[str, Any]) -> Union[Dict, Any]:
    """
    Rewrap dict data into the original GraphState type if needed.

    Args:
        state_type: The original type of the state object
        data: The dict data to wrap

    Returns:
        Instance of state_type if it's a Pydantic model, otherwise the dict
    """
    try:
        # If state_type is a pydantic BaseModel class, instantiate it
        if hasattr(state_type, "__bases__") and any(
            "BaseModel" in str(base) for base in state_type.__mro__
        ):
            return state_type(**data)
        # If it's a dict type or the graph accepts dicts, return dict
        return data
    except Exception as e:
        # Fall back to dict if instantiation fails
        logger.warning(
            f"Could not instantiate {state_type} with data, falling back to dict: {e}"
        )
        return data


class BaseNodeHandler(ABC):
    """Base class for all LangGraph node handlers with common wrapper logic."""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logging.getLogger(f"{__name__}.{node_name}")

    @abstractmethod
    async def execute(
        self, state: Dict[str, Any], config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute the node-specific logic. Must be implemented by subclasses."""
        pass

    async def __call__(
        self, state: Union[Dict, Any], *, config: Optional[Dict] = None, store=None
    ) -> Union[Dict, Any]:
        """
        Common wrapper logic for all nodes - eliminates repetition.

        Handles:
        - Error logging and propagation
        - Execution timing
        - State validation
        - Workflow path tracking
        - State type normalization (Pydantic <-> dict)
        """
        start_time = time.time()
        incoming_type = type(state)

        # Convert to dict for processing
        s = to_state_dict(state)

        # Add node to workflow path
        workflow_path = s.get("workflow_path", [])
        s = {**s, "workflow_path": workflow_path + [self.node_name]}

        try:
            # Log node entry
            log_event(
                stage=self.node_name,
                event="start",
                query=s.get("normalized_query", ""),
                thread_id=config.get("configurable", {}).get("thread_id")
                if config
                else None,
            )

            # Execute node-specific logic - pass normalized dict and config
            result = await self.execute(s, config)

            # Ensure result is a dict
            if not isinstance(result, dict):
                raise ValueError(
                    f"Node {self.node_name} must return a dictionary, got {type(result)}"
                )

            # Log success
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="success",
                ms=execution_time,
                result_count=len(result.get("search_results", []))
                if "search_results" in result
                else 0,
            )

            # Merge result with existing state to preserve all fields
            merged = {**s, **result}
            return from_state_dict(incoming_type, merged)

        except Exception as e:
            # Log error
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                ms=execution_time,
            )

            # Add to error messages for debugging
            error_messages = s.get("error_messages", [])
            error_message = f"Error in {self.node_name}: {str(e)}"

            self.logger.error(error_message, exc_info=True)

            merged = {**s, "error_messages": error_messages + [error_message]}
            return from_state_dict(incoming_type, merged)


class SearchNodeHandler(BaseNodeHandler):
    """Base class for search-related nodes with search-specific logging."""

    async def __call__(
        self, state: Union[Dict, Any], *, config: Optional[Dict] = None, store=None
    ) -> Union[Dict, Any]:
        """Enhanced wrapper for search nodes with search-specific metrics."""
        start_time = time.time()
        incoming_type = type(state)

        # Convert to dict for processing
        s = to_state_dict(state)

        # Add node to workflow path
        workflow_path = s.get("workflow_path", [])
        s = {**s, "workflow_path": workflow_path + [self.node_name]}

        try:
            # Log search node entry with query details
            log_event(
                stage=self.node_name,
                event="start",
                query=s.get("normalized_query", ""),
                intent=get_intent_label(s.get("intent")) if s.get("intent") else None,
                thread_id=config.get("configurable", {}).get("thread_id")
                if config
                else None,
            )

            # Execute search logic - pass normalized dict and config
            result = await self.execute(s, config)

            # Enhanced search logging
            execution_time = (time.time() - start_time) * 1000
            search_results = result.get("search_results", [])

            log_event(
                stage=self.node_name,
                event="success",
                ms=execution_time,
                result_count=len(search_results),
                avg_score=sum(r.score for r in search_results) / len(search_results)
                if search_results
                else 0,
                top_score=max(r.score for r in search_results) if search_results else 0,
            )

            # Merge result with existing state to preserve all fields
            merged = {**s, **result}
            return from_state_dict(incoming_type, merged)

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                ms=execution_time,
            )

            error_messages = s.get("error_messages", [])
            error_message = f"Search error in {self.node_name}: {str(e)}"
            self.logger.error(error_message, exc_info=True)

            merged = {
                **s,
                "error_messages": error_messages + [error_message],
                "search_results": [],  # Provide empty results for search failures
            }
            return from_state_dict(incoming_type, merged)
