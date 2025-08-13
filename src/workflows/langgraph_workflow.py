"""Main LangGraph workflow implementation for multi-step retrieval."""

import logging
from typing import Dict, Any, AsyncGenerator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from workflows.state import WorkflowState, WorkflowConfig, initialize_workflow_state
from workflows.agents import (
    query_decomposer_agent,
    search_orchestrator_agent, 
    single_search_agent,
    parallel_search_agent,
    result_synthesizer_agent
)
from workflows.response_generator import response_generator_agent
from infra.resource_manager import RAGResources
from services.models import TurnResult, IntentResult

logger = logging.getLogger(__name__)


class LangGraphWorkflow:
    """
    LangGraph-based workflow for sophisticated query processing.
    
    This workflow replaces the monolithic turn controller with a multi-agent system
    that can handle complex queries through decomposition and parallel search.
    """
    
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        self.checkpointer = MemorySaver()  # For conversation state persistence
        self._workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all agents and routing logic."""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add agent nodes
        workflow.add_node("query_decomposer", self._wrap_agent(query_decomposer_agent))
        workflow.add_node("search_orchestrator", self._wrap_agent(search_orchestrator_agent))
        workflow.add_node("single_search_agent", self._wrap_agent(single_search_agent))
        workflow.add_node("parallel_search_agent", self._wrap_agent(parallel_search_agent))
        workflow.add_node("result_synthesizer", self._wrap_agent(result_synthesizer_agent))
        workflow.add_node("response_generator", self._wrap_agent(response_generator_agent))
        workflow.add_node("error_handler", self._error_handler)
        
        # Define workflow edges
        workflow.add_edge(START, "query_decomposer")
        workflow.add_edge("query_decomposer", "search_orchestrator")
        
        # Search orchestrator routes to either single or parallel search
        # (routing is handled by the Command returns in search_orchestrator_agent)
        
        # Both search paths lead to result synthesis
        workflow.add_edge("single_search_agent", "result_synthesizer")
        workflow.add_edge("parallel_search_agent", "result_synthesizer")
        
        # Synthesis leads to response generation
        workflow.add_edge("result_synthesizer", "response_generator")
        
        # Response generation leads to end
        workflow.add_edge("response_generator", END)
        
        # Error handling
        workflow.add_edge("error_handler", END)
        
        # Compile the workflow
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _wrap_agent(self, agent_func):
        """Wrap agent function to provide resources and handle errors."""
        async def wrapped_agent(state: WorkflowState):
            try:
                # Get resources from context (would be injected in real implementation)
                resources = state.get("_resources")
                if not resources:
                    raise RuntimeError("Resources not available in workflow state")
                
                # Call the agent
                result = await agent_func(state, resources)
                return result
                
            except Exception as e:
                error_msg = f"Agent {agent_func.__name__} failed: {e}"
                logger.error(error_msg)
                return {
                    "error_messages": [error_msg],
                    "workflow_path": [f"{agent_func.__name__}_error"]
                }
        
        return wrapped_agent
    
    async def _error_handler(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle workflow errors gracefully."""
        errors = state.get("error_messages", [])
        error_summary = "; ".join(errors[-3:])  # Last 3 errors
        
        return {
            "final_answer": f"I encountered an error while processing your request: {error_summary}",
            "workflow_path": ["error_handler"]
        }
    
    async def stream(self, user_input: str, resources: RAGResources, 
                    request_id: str = None, user_context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the workflow execution with progress updates.
        
        This method provides the same interface as the original handle_turn function
        but uses the LangGraph workflow internally.
        """
        try:
            # Initialize workflow state
            initial_state = initialize_workflow_state(
                query=user_input,
                request_id=request_id,
                user_context=user_context
            )
            
            # Inject resources into state (temporary approach)
            initial_state["_resources"] = resources
            
            # Configure workflow execution
            config = {
                "configurable": {"thread_id": request_id or "default"},
                "recursion_limit": 50
            }
            
            # Stream workflow execution
            async for chunk in self._workflow.astream(initial_state, config=config):
                # Process and yield updates
                for node_name, node_update in chunk.items():
                    if node_name == END:
                        # Final result
                        final_state = node_update
                        yield self._format_final_result(final_state, request_id)
                    else:
                        # Intermediate progress
                        yield self._format_progress_update(node_name, node_update, request_id)
                        
                        # If this is response generation, stream the response chunks
                        if node_name == "response_generator" and "response_chunks" in node_update:
                            for chunk_text in node_update["response_chunks"]:
                                yield {
                                    "type": "response_chunk",
                                    "content": chunk_text,
                                    "turn_id": f"workflow_{request_id}",
                                    "req_id": request_id
                                }
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            yield {
                "type": "error",
                "result": {
                    "answer": f"Workflow execution failed: {str(e)}",
                    "sources": [],
                    "intent": IntentResult(intent="error", confidence=0.0),
                    "response_time_ms": 0,
                    "error": str(e)
                },
                "turn_id": f"workflow_{request_id}",
                "req_id": request_id
            }
    
    def _format_progress_update(self, node_name: str, node_update: Dict[str, Any], 
                              request_id: str) -> Dict[str, Any]:
        """Format intermediate progress updates."""
        # Map node names to user-friendly status messages
        status_messages = {
            "query_decomposer": "Analyzing query complexity...",
            "search_orchestrator": "Planning search strategy...", 
            "single_search_agent": "Searching knowledge base...",
            "parallel_search_agent": "Performing parallel searches...",
            "result_synthesizer": "Synthesizing results...",
            "response_generator": "Generating response...",
            "error_handler": "Handling error..."
        }
        
        return {
            "type": "status",
            "message": status_messages.get(node_name, f"Processing {node_name}..."),
            "turn_id": f"workflow_{request_id}",
            "req_id": request_id,
            "node": node_name,
            "details": {
                "workflow_path": node_update.get("workflow_path", []),
                "metrics": node_update.get("performance_metrics", {})
            }
        }
    
    def _format_final_result(self, final_state: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Format the final workflow result."""
        # Extract sources from search results
        search_results = final_state.get("search_results", [])
        sources = []
        for result in search_results:
            sources.append({
                "title": result.metadata.get("title", "Document"),
                "url": result.metadata.get("url", "#"),
                "score": result.score
            })
        
        # Create TurnResult-compatible structure
        turn_result = {
            "answer": final_state.get("final_answer", "No answer generated."),
            "sources": sources[:10],  # Limit to top 10 sources
            "intent": final_state.get("intent", IntentResult(intent="unknown", confidence=0.0)).dict(),
            "response_time_ms": sum(final_state.get("performance_metrics", {}).values()),
            "workflow_path": final_state.get("workflow_path", []),
            "metrics": final_state.get("performance_metrics", {})
        }
        
        return {
            "type": "complete",
            "result": turn_result,
            "turn_id": f"workflow_{request_id}",
            "req_id": request_id
        }


# Factory function for creating workflow instances
def create_langgraph_workflow(config: WorkflowConfig = None) -> LangGraphWorkflow:
    """Create a new LangGraph workflow instance."""
    return LangGraphWorkflow(config)


# Integration function to replace handle_turn
async def handle_turn_with_langgraph(
    user_input: str,
    resources: RAGResources,
    chat_history: list = None,
    use_mock_corpus: bool = False,
    workflow_config: WorkflowConfig = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    LangGraph-based replacement for the original handle_turn function.
    
    This provides the same interface but uses the sophisticated multi-agent workflow.
    """
    import time
    
    # Create workflow instance
    workflow = create_langgraph_workflow(workflow_config)
    
    # Generate request ID
    request_id = f"lg_{int(time.time() * 1000)}"
    
    # Prepare user context
    user_context = {
        "use_mock_corpus": use_mock_corpus,
        "chat_history": chat_history or []
    }
    
    # Stream workflow execution
    async for update in workflow.stream(
        user_input=user_input,
        resources=resources,
        request_id=request_id,
        user_context=user_context
    ):
        yield update