"""Orchestrator node for conversational RAG with tool use."""

import logging
import os
from typing import Dict, Any, Optional

from .base_node import to_state_dict, from_state_dict

logger = logging.getLogger(__name__)


async def orchestrator_node(state, config=None, *, store=None):
    """
    Orchestrator node that plans and executes multi-step RAG workflows.
    
    This node integrates the LLM orchestrator for conversational search
    when enabled via feature flag.
    
    Args:
        state: Workflow state (GraphState or dict)
        config: RunnableConfig with user context
        store: Optional BaseStore for cross-thread memory
        
    Returns:
        State update with orchestrated results
    """
    incoming_type = type(state)
    s = to_state_dict(state)
    
    try:
        # Check if orchestration is enabled
        use_orchestrator = os.getenv("ENABLE_ORCHESTRATOR", "false").lower() == "true"
        
        if not use_orchestrator:
            # Fall back to standard search flow
            logger.info("Orchestrator disabled, routing to standard search")
            return from_state_dict(incoming_type, {
                **s,
                "workflow_path": s.get("workflow_path", []) + ["orchestrator_bypass"],
            })
        
        # Import orchestrator
        from src.agent.orchestrator import LLMOrchestrator, orchestrated_search
        from src.infra.resource_manager import get_resources
        
        # Get query and context
        query = s.get("normalized_query", s.get("original_query", ""))
        chat_history = s.get("chat_history", [])
        user_context = s.get("user_context", {})
        
        logger.info(f"Orchestrating query: {query[:100]}...")
        
        # Get resources for tool execution
        resources = get_resources()
        
        # Run orchestrated search
        results = await orchestrated_search(
            query=query,
            resources=resources,
            context=chat_history,
            use_orchestrator=True
        )
        
        # Update state with orchestrated results
        return from_state_dict(incoming_type, {
            **s,
            "search_results": results.get("search_results", []),
            "final_answer": results.get("final_answer"),
            "tool_outputs": results.get("tool_outputs", []),
            "workflow_path": s.get("workflow_path", []) + ["orchestrator"],
            "orchestrator_metadata": {
                "tools_used": len(results.get("tool_outputs", [])),
                "search_count": len(results.get("search_results", [])),
            }
        })
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Fall back to standard flow on error
        return from_state_dict(incoming_type, {
            **s,
            "workflow_path": s.get("workflow_path", []) + ["orchestrator_error"],
            "error_messages": s.get("error_messages", []) + [f"Orchestrator error: {e}"],
        })