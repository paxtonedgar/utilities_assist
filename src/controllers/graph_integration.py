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

# Import state key constants from centralized location
from agent.constants import ORIGINAL_QUERY, NORMALIZED_QUERY, INTENT

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
        thread_id: Optional thread ID for conversation persistence
        user_context: Optional user context from authentication
        
    Yields:
        Dict with turn progress updates and final result
    """
    # Generate request ID and set context at the start
    req_id = generate_req_id()
    set_context_var("current_req_id", req_id)
    
    # DEBUG: Log exactly what we received
    logger.error(f"DEBUG handle_turn received user_input: '{repr(user_input)}' (type: {type(user_input)}, len: {len(user_input) if user_input else 'None'})")
    
    # Normalize and validate user input to prevent corrupted queries
    if user_input is None:
        user_input = ""
    
    # Handle escaped/corrupted input - common issue from UI layers
    if isinstance(user_input, str):
        # Fix escaped backslashes and other common corruptions
        user_input = user_input.replace('\\\\', '').replace("\\'", "'").strip()
        # Remove any null bytes or control characters
        user_input = ''.join(char for char in user_input if ord(char) >= 32 or char in '\t\n\r')
    
    # Validate after normalization
    if not user_input or not user_input.strip():
        logger.error(f"Empty or invalid user input after normalization: original='{repr(user_input)}'")
        yield {
            "type": "error",
            "message": "Please provide a valid query. Empty or corrupted queries are not supported.",
            "req_id": req_id
        }
        return
    
    # DEBUG: Log normalized input
    logger.error(f"DEBUG after normalization user_input: '{repr(user_input)}' (len: {len(user_input)})")
    
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
    
    # Additional validation at graph level
    if not user_input or not user_input.strip():
        logger.error(f"Graph received empty user input: '{repr(user_input)}'")
        yield {
            "type": "error",
            "message": "Invalid query: empty or whitespace-only input not allowed",
            "turn_id": turn_id,
            "req_id": req_id
        }
        return
    
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
        
        # Sanitize input once
        text = (user_input or "").strip()
        
        # DEBUG: Log state creation
        logger.error(f"DEBUG creating initial_state with user_input: '{repr(user_input)}' sanitized: '{repr(text)}'")
        
        # Initialize graph state as plain dict - NO GraphState construction
        # This prevents "GraphState object has no attribute 'get'" errors
        initial_state = {
            ORIGINAL_QUERY: text,                 # Required by summarize_node 
            NORMALIZED_QUERY: text,               # Start normalized as original
            INTENT: None,
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
            "rewrite_attempts": 0,  # Track rewrite attempts to prevent infinite loops
            "coverage_threshold": 0.7,
            "min_results": 3,
            "error_messages": [],
            "_use_mock_corpus": False  # CRITICAL: Turn mocks OFF - use real OpenSearch
        }
        
        # Guardrails: Verify state is dict-like before passing to LangGraph
        if not isinstance(initial_state, dict):
            logger.error("Initial state is not a dict-like mapping: %r", type(initial_state))
            raise TypeError(f"Expected dict-like initial state, got {type(initial_state)}")
        
        # State sanity log - confirm keys are present (extended with key list)
        logger.info(
            "TURN_START user_id=%s thread_id=%s | keys=%s | original=%r normalized=%r",
            user_context.get("user_id", "unknown"), thread_id,
            sorted(list(initial_state.keys()))[:10],  # First 10 keys for sanity
            initial_state.get(ORIGINAL_QUERY, "")[:80],
            initial_state.get(NORMALIZED_QUERY, "")[:80]
        )
        
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


def _format_graph_final_result(final_state, start_time: float, turn_id: str, req_id: str) -> Dict[str, Any]:
    """Format final graph state as TurnResult compatible response with missing features."""
    
    # CRITICAL: Convert GraphState to dict at graph output boundary
    # This fixes the "GraphState object has no attribute 'get'" error
    if hasattr(final_state, 'model_dump'):
        # Pydantic v2
        state_dict = final_state.model_dump()
    elif hasattr(final_state, 'dict'):
        # Pydantic v1
        state_dict = final_state.dict()
    elif isinstance(final_state, dict):
        # Already a dict
        state_dict = final_state
    else:
        # Last resort - attempt dict conversion
        state_dict = dict(final_state)
    
    # Extract results from normalized dict
    final_answer = state_dict.get("final_answer", "No answer generated.")
    combined_results = state_dict.get("combined_results", [])
    workflow_path = state_dict.get("workflow_path", [])
    
    # MISSING: Use source chips if available (from traditional pipeline)
    source_chips = state_dict.get("source_chips", [])
    if source_chips:
        sources = source_chips  # Use extracted source chips
    else:
        # Fallback: Build sources from combined results with required fields
        sources = []
        for result in combined_results[:10]:  # Limit to top 10
            # Extract real URL or construct one from metadata
            url = result.metadata.get("url") or result.metadata.get("page_url") or result.metadata.get("link")
            if not url or url == "#":
                # Construct a meaningful URL or use a placeholder
                url = f"#doc-{result.doc_id}" if result.doc_id else "#"
            
            sources.append({
                "doc_id": result.doc_id,  # Required field
                "title": result.metadata.get("title") or result.metadata.get("page_title") or "Document",
                "url": url,
                "excerpt": result.content[:200] + "..." if len(result.content) > 200 else result.content
            })
    
    # MISSING: Include verification metrics (from traditional pipeline)
    verification_metrics = state_dict.get("verification_metrics", {})
    
    # Handle intent - convert dict to IntentResult if needed
    intent_data = state_dict.get("intent", {"intent": "unknown", "confidence": 0.0})
    if isinstance(intent_data, dict):
        intent_result = IntentResult(
            intent=intent_data.get("intent", "unknown"),
            confidence=intent_data.get("confidence", 0.0)
        )
    else:
        intent_result = intent_data  # Already an IntentResult
    
    # Validate sources before TurnResult construction
    for i, source in enumerate(sources):
        if not source.get("doc_id"):
            logger.warning(f"Source {i} missing doc_id: {source}")
            source["doc_id"] = f"unknown-{i}"  # Fallback
    
    # Create TurnResult-compatible structure with missing features
    turn_result = TurnResult(
        answer=final_answer,
        sources=sources,
        intent=intent_result,
        response_time_ms=(time.time() - start_time) * 1000,
        graph_workflow_path=workflow_path,  # Additional field for graph tracking
        graph_loop_count=state_dict.get("loop_count", 0),
        verification=verification_metrics  # MISSING: Answer quality verification
    )
    
    # MISSING: Stream verification metrics (from traditional pipeline)
    result_dict = turn_result.model_dump()  # Use model_dump instead of deprecated dict()
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