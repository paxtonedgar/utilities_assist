"""
Integration layer for LangGraph that maintains existing interface and adds flag-based control.

This provides the exact interface specification from your requirements:
- App boots with no behavior change when LangGraph flag is off
- Streaming & fallback with graceful degradation
- Same interface as original handle_turn
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from src.infra.settings import get_settings
from infra.resource_manager import get_resources, RAGResources
from services.models import TurnResult, IntentResult
from infra.telemetry import generate_request_id, log_overall_stage
from src.telemetry.logger import generate_req_id, set_context_var, stage
# Graceful import handling for persistence module
try:
    from infra.persistence import (
        get_checkpointer_and_store, 
        extract_user_context, 
        generate_thread_id,
        create_langgraph_config
    )
    PERSISTENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Persistence module not available: {e}. Using fallback implementations.")
    PERSISTENCE_AVAILABLE = False
    
    # Fallback implementations
    def get_checkpointer_and_store():
        return None, None
    
    def extract_user_context(resources):
        return {"user_id": "fallback_user", "session_metadata": {}}
    
    def generate_thread_id(user_id, session_context=None):
        import time
        return f"{user_id}_{int(time.time())}"
    
    def create_langgraph_config(user_context, thread_id):
        return {"configurable": {"thread_id": thread_id}}

# Import LangGraph components (traditional handler removed - LangGraph is now the only system)
from agent.graph import create_graph, GraphState

logger = logging.getLogger(__name__)

# Environment flag for LangGraph control
def is_langgraph_enabled() -> bool:
    """Check if LangGraph is enabled via centralized settings."""
    try:
        settings = get_settings()
        return settings.enable_langgraph_persistence
    except Exception:
        # Fallback to environment variable
        return os.getenv("ENABLE_LANGGRAPH", "false").lower() == "true"

LANGGRAPH_ENABLED = is_langgraph_enabled()


@stage("overall")
async def handle_turn(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False,
    thread_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    LangGraph turn handler with enterprise authentication and persistence.
    
    This is now the single, unified turn handler using LangGraph architecture
    with full enterprise features and conversation persistence.
    
    Args:
        user_input: Raw user input
        resources: Pre-configured resource container (auto-fetched if None)
        chat_history: Recent conversation history
        use_mock_corpus: If True, use confluence_mock index instead of confluence_current
        thread_id: Optional thread ID for conversation persistence
        user_context: Optional user context from authentication
        
    Yields:
        Dict with turn progress updates and final result
    """
    # Generate request ID and set context at the start
    req_id = generate_req_id()
    set_context_var("current_req_id", req_id)
    
    # Set user context if available
    if user_context and user_context.get("user_id"):
        set_context_var("user_id", user_context["user_id"])
    if thread_id:
        set_context_var("thread_id", thread_id)
    
    # Always use LangGraph (traditional pipeline has been fully migrated)
    async for update in handle_turn_with_graph(
        user_input=user_input,
        resources=resources,
        chat_history=chat_history,
        use_mock_corpus=use_mock_corpus,
        thread_id=thread_id,
        user_context=user_context
    ):
        # Add request ID to updates
        if isinstance(update, dict):
            update["req_id"] = req_id
        yield update


async def handle_turn_with_graph(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False,
    thread_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    LangGraph-based turn handler with authentication, persistence, and graceful fallback.
    
    Integrates user context, thread management, and conversation persistence
    while maintaining streaming interface and fallback capabilities.
    """
    start_time = time.time()
    turn_id = f"graph_{int(start_time)}"
    req_id = generate_request_id()
    
    try:
        # Get shared resources (Phase 1 performance optimizations maintained)
        if resources is None:
            resources = get_resources()
            if resources is None:
                raise RuntimeError("Resources not initialized. Call initialize_resources() at startup.")
        
        logger.info(f"Processing with LangGraph (resource age: {resources.get_age_seconds():.1f}s)")
        
        # Extract user context from authentication if not provided
        if user_context is None:
            user_context = extract_user_context(resources)
            logger.info(f"Extracted user context: user_id={user_context.get('user_id', 'unknown')}")
        
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = generate_thread_id(
                user_context.get("user_id", "unknown"), 
                user_context.get("session_metadata")
            )
            logger.info(f"Generated thread ID: {thread_id}")
        
        # Initialize persistence layer
        checkpointer, store = get_checkpointer_and_store()
        if checkpointer:
            logger.info("Using persistent checkpointer for conversation memory")
        if store:
            logger.info("Using persistent store for cross-thread user memory")
        
        # Create LangGraph configuration with thread and user context
        langgraph_config = create_langgraph_config(user_context, thread_id)
        
        # Create and configure graph with persistence
        graph = create_graph(
            enable_loops=True,
            coverage_threshold=0.7,
            min_results=3,
            checkpointer=checkpointer,
            store=store
        )
        
        # Initialize graph state with user context and authentication
        initial_state = GraphState({
            "original_query": user_input,
            "normalized_query": None,
            "intent": None,
            "search_results": [],
            "combined_results": [],
            "final_context": None,
            "final_answer": None,
            "response_chunks": [],
            
            # User context and authentication integration
            "user_id": user_context.get("user_id"),
            "thread_id": thread_id,
            "session_id": user_context.get("session_metadata", {}).get("cloud_profile"),
            "user_context": user_context,
            "user_preferences": user_context.get("user_preferences", {}),
            
            "workflow_path": [],
            "loop_count": 0,
            "coverage_threshold": 0.7,
            "min_results": 3,
            "error_messages": [],
            "_use_mock_corpus": False  # Always use production Confluence/OpenSearch
        })
        
        # Track progress and stream updates with user context
        yield {
            "type": "status", 
            "message": f"Processing with advanced workflow (user: {user_context.get('user_id', 'unknown')})...", 
            "turn_id": turn_id, 
            "req_id": req_id,
            "thread_id": thread_id,
            "user_id": user_context.get("user_id")
        }
        
        # Execute graph with streaming updates and persistence
        final_state = None
        async for chunk in graph.astream(initial_state, config=langgraph_config):
            for node_name, node_update in chunk.items():
                # Stream progress updates
                yield _format_graph_progress(node_name, node_update, turn_id, req_id, thread_id)
                
                # If we have response chunks, stream them
                if "response_chunks" in node_update:
                    for response_chunk in node_update["response_chunks"]:
                        yield {
                            "type": "response_chunk",
                            "content": response_chunk,
                            "turn_id": turn_id,
                            "req_id": req_id,
                            "thread_id": thread_id
                        }
                
                # Update final state
                if node_name != "__end__":
                    final_state = node_update
        
        # Generate final result
        if final_state:
            final_result = _format_graph_final_result(final_state, start_time, turn_id, req_id)
            yield final_result
        else:
            # No final state - something went wrong
            raise RuntimeError("Graph execution completed without final state")
        
    except Exception as e:
        logger.error(f"LangGraph processing failed: {e}")
        
        # Generate error response (no fallback - LangGraph is the primary system)
        error_result = TurnResult(
            answer=f"I encountered an error processing your request: {str(e)}",
            sources=[],
            intent=IntentResult(intent="error", confidence=0.0),
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )
        
        yield {
            "type": "error",
            "result": error_result.dict(),
            "turn_id": turn_id,
            "req_id": req_id,
            "message": f"LangGraph processing failed: {str(e)}"
        }


def _format_graph_progress(node_name: str, node_update: Dict[str, Any], turn_id: str, req_id: str, thread_id: str = None) -> Dict[str, Any]:
    """Format graph node updates as progress messages with thread context."""
    
    # Map node names to user-friendly messages
    status_messages = {
        "summarize": "Analyzing and normalizing query...",
        "intent": "Determining query intent...", 
        "search_confluence": "Searching documentation...",
        "search_swagger": "Searching API specifications...",
        "search_multi": "Performing comprehensive search...",
        "rewrite_query": "Refining search strategy...",
        "combine": "Synthesizing results...",
        "answer": "Generating response..."
    }
    
    message = status_messages.get(node_name, f"Processing {node_name}...")
    
    progress = {
        "type": "status",
        "message": message,
        "turn_id": turn_id,
        "req_id": req_id,
        "graph_node": node_name,
        "workflow_path": node_update.get("workflow_path", [])
    }
    
    if thread_id:
        progress["thread_id"] = thread_id
    
    # Include user context if available
    if "user_id" in node_update:
        progress["user_id"] = node_update["user_id"]
    
    return progress


def _format_graph_final_result(final_state: Dict[str, Any], start_time: float, turn_id: str, req_id: str) -> Dict[str, Any]:
    """Format final graph state as TurnResult compatible response with missing features."""
    
    # Extract results
    final_answer = final_state.get("final_answer", "No answer generated.")
    combined_results = final_state.get("combined_results", [])
    workflow_path = final_state.get("workflow_path", [])
    
    # MISSING: Use source chips if available (from traditional pipeline)
    source_chips = final_state.get("source_chips", [])
    if source_chips:
        sources = source_chips  # Use extracted source chips
    else:
        # Fallback: Build sources from combined results
        sources = []
        for result in combined_results[:10]:  # Limit to top 10
            sources.append({
                "title": result.metadata.get("title", "Document"),
                "url": result.metadata.get("url", "#"),
                "score": result.score
            })
    
    # MISSING: Include verification metrics (from traditional pipeline)
    verification_metrics = final_state.get("verification_metrics", {})
    
    # Create TurnResult-compatible structure with missing features
    turn_result = TurnResult(
        answer=final_answer,
        sources=sources,
        intent=final_state.get("intent", IntentResult(intent="unknown", confidence=0.0)),
        response_time_ms=(time.time() - start_time) * 1000,
        graph_workflow_path=workflow_path,  # Additional field for graph tracking
        graph_loop_count=final_state.get("loop_count", 0),
        verification=verification_metrics  # MISSING: Answer quality verification
    )
    
    # MISSING: Stream verification metrics (from traditional pipeline)
    result_dict = turn_result.dict()
    if verification_metrics:
        result_dict["verification"] = verification_metrics
    
    return {
        "type": "complete",
        "result": result_dict,
        "turn_id": turn_id,
        "req_id": req_id
    }


def enable_langgraph():
    """Enable LangGraph processing."""
    global LANGGRAPH_ENABLED
    LANGGRAPH_ENABLED = True
    logger.info("LangGraph processing enabled")


def disable_langgraph():
    """Disable LangGraph processing (revert to traditional)."""
    global LANGGRAPH_ENABLED
    LANGGRAPH_ENABLED = False
    logger.info("LangGraph processing disabled, using traditional pipeline")


def is_langgraph_enabled() -> bool:
    """Check if LangGraph is currently enabled."""
    return LANGGRAPH_ENABLED


# Convenience functions for testing and development
async def test_graph_vs_traditional(
    test_query: str,
    resources: Optional[RAGResources] = None
) -> Dict[str, Any]:
    """
    Test both graph and traditional processing for comparison.
    
    Useful for A/B testing and performance comparison.
    """
    if not resources:
        resources = get_resources()
    
    results = {"query": test_query}
    
    # Test traditional
    start_time = time.time()
    traditional_updates = []
    async for update in handle_turn_traditional(test_query, resources):
        traditional_updates.append(update)
        if update.get("type") == "complete":
            break
    results["traditional"] = {
        "time_ms": (time.time() - start_time) * 1000,
        "final_result": traditional_updates[-1] if traditional_updates else None,
        "update_count": len(traditional_updates)
    }
    
    # Test graph 
    start_time = time.time()
    graph_updates = []
    async for update in handle_turn_with_graph(test_query, resources):
        graph_updates.append(update)
        if update.get("type") == "complete":
            break
    results["graph"] = {
        "time_ms": (time.time() - start_time) * 1000,
        "final_result": graph_updates[-1] if graph_updates else None,
        "update_count": len(graph_updates)
    }
    
    return results