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
from infra.resource_manager import get_resources, RAGResources
from services.models import TurnResult, IntentResult
from infra.telemetry import generate_request_id, log_overall_stage

# Import both traditional and graph-based handlers
from controllers.turn_controller import handle_turn as handle_turn_traditional
from agent.graph import create_graph, GraphState

logger = logging.getLogger(__name__)

# Environment flag for LangGraph control
LANGGRAPH_ENABLED = os.getenv("ENABLE_LANGGRAPH", "false").lower() == "true"


async def handle_turn(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Enhanced turn handler with LangGraph integration.
    
    This function maintains the exact same interface as the original handle_turn
    but can switch to LangGraph processing based on environment flag.
    
    Args:
        user_input: Raw user input
        resources: Pre-configured resource container (auto-fetched if None)
        chat_history: Recent conversation history
        use_mock_corpus: If True, use confluence_mock index instead of confluence_current
        
    Yields:
        Dict with turn progress updates and final result
    """
    if LANGGRAPH_ENABLED:
        # Use LangGraph workflow
        async for update in handle_turn_with_graph(
            user_input=user_input,
            resources=resources,
            chat_history=chat_history,
            use_mock_corpus=use_mock_corpus
        ):
            yield update
    else:
        # Use traditional pipeline (no behavior change)
        async for update in handle_turn_traditional(
            user_input=user_input,
            resources=resources,
            chat_history=chat_history,
            use_mock_corpus=use_mock_corpus
        ):
            yield update


async def handle_turn_with_graph(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    LangGraph-based turn handler with fallback to traditional processing.
    
    Implements the A4 requirement: Keep streaming, add graceful fallback.
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
        
        # Create and configure graph
        graph = create_graph(
            enable_loops=True,
            coverage_threshold=0.7,
            min_results=3
        )
        
        # Initialize graph state
        initial_state = GraphState({
            "original_query": user_input,
            "normalized_query": None,
            "intent": None,
            "search_results": [],
            "combined_results": [],
            "final_context": None,
            "final_answer": None,
            "response_chunks": [],
            "workflow_path": [],
            "loop_count": 0,
            "coverage_threshold": 0.7,
            "min_results": 3,
            "error_messages": [],
            "_resources": resources,
            "_use_mock_corpus": use_mock_corpus
        })
        
        # Track progress and stream updates
        yield {"type": "status", "message": "Processing with advanced workflow...", "turn_id": turn_id, "req_id": req_id}
        
        # Execute graph with streaming updates
        final_state = None
        async for chunk in graph.astream(initial_state):
            for node_name, node_update in chunk.items():
                # Stream progress updates
                yield _format_graph_progress(node_name, node_update, turn_id, req_id)
                
                # If we have response chunks, stream them
                if "response_chunks" in node_update:
                    for response_chunk in node_update["response_chunks"]:
                        yield {
                            "type": "response_chunk",
                            "content": response_chunk,
                            "turn_id": turn_id,
                            "req_id": req_id
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
        
        # A4 Requirement: Graceful fallback to traditional processing
        logger.info("Falling back to traditional pipeline...")
        
        yield {
            "type": "status", 
            "message": "Switching to standard processing...", 
            "turn_id": turn_id,
            "req_id": req_id
        }
        
        # Stream fallback processing
        async for update in handle_turn_traditional(
            user_input=user_input,
            resources=resources,
            chat_history=chat_history,
            use_mock_corpus=use_mock_corpus
        ):
            # Mark fallback updates
            if isinstance(update, dict):
                update["fallback"] = True
                update["original_error"] = str(e)
            yield update


def _format_graph_progress(node_name: str, node_update: Dict[str, Any], turn_id: str, req_id: str) -> Dict[str, Any]:
    """Format graph node updates as progress messages."""
    
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
    
    return {
        "type": "status",
        "message": message,
        "turn_id": turn_id,
        "req_id": req_id,
        "graph_node": node_name,
        "workflow_path": node_update.get("workflow_path", [])
    }


def _format_graph_final_result(final_state: Dict[str, Any], start_time: float, turn_id: str, req_id: str) -> Dict[str, Any]:
    """Format final graph state as TurnResult compatible response."""
    
    # Extract results
    final_answer = final_state.get("final_answer", "No answer generated.")
    combined_results = final_state.get("combined_results", [])
    workflow_path = final_state.get("workflow_path", [])
    
    # Build sources from combined results
    sources = []
    for result in combined_results[:10]:  # Limit to top 10
        sources.append({
            "title": result.metadata.get("title", "Document"),
            "url": result.metadata.get("url", "#"),
            "score": result.score
        })
    
    # Create TurnResult-compatible structure
    turn_result = TurnResult(
        answer=final_answer,
        sources=sources,
        intent=final_state.get("intent", IntentResult(intent="unknown", confidence=0.0)),
        response_time_ms=(time.time() - start_time) * 1000,
        graph_workflow_path=workflow_path,  # Additional field for graph tracking
        graph_loop_count=final_state.get("loop_count", 0)
    )
    
    return {
        "type": "complete",
        "result": turn_result.dict(),
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