# src/agent/nodes/intent.py
"""
Legacy Intent Node: Simplified stub for backward compatibility.

The new architecture uses micro-router instead of complex intent classification.
This node exists only for backward compatibility with existing LangGraph flows.
"""

import logging
from typing import Dict, Any

from .base_node import BaseNodeHandler

logger = logging.getLogger(__name__)


class LegacyIntentNode(BaseNodeHandler):
    """Legacy intent node for backward compatibility."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """
        Legacy intent classification - now just passes through.
        
        The new architecture handles routing in the unified_router node using
        micro-router patterns instead of complex intent classification.
        """
        query = state.get("normalized_query", state.get("original_query", ""))
        
        # Create minimal intent object for compatibility
        legacy_intent = {
            "intent": "general",  # Default to general
            "confidence": 0.5,    # Medium confidence
            "method": "legacy_stub"
        }
        
        logger.info(f"Legacy intent node (stub): '{query}' -> general (micro-router handles routing)")
        
        return {
            **state,
            "intent": legacy_intent,
            "workflow_path": state.get("workflow_path", []) + ["intent_legacy"]
        }


# Legacy function for backward compatibility
async def intent_node(state, config, *, store=None):
    """Legacy intent_node function for backward compatibility."""
    node = LegacyIntentNode()
    return await node.execute(state, config)


# Export for backward compatibility
__all__ = ["intent_node", "LegacyIntentNode"]