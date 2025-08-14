# src/agent/nodes/processing_nodes.py
"""
Processing node handlers for combining results, generating answers, etc.

Clean implementations using base node pattern.
"""

from typing import Dict, Any, List
import logging

from .base_node import BaseNodeHandler
from agent.nodes.combine import combine_node
from services.respond import generate_response, extract_source_chips, verify_answer
from services.models import SearchResult

logger = logging.getLogger(__name__)


class CombineNode(BaseNodeHandler):
    """Handles combining and ranking search results."""
    
    def __init__(self):
        super().__init__("combine")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute result combination logic."""
        search_results = state.get("search_results", [])
        intent = state.get("intent")
        
        # Use existing combine_node function
        combined_result = await combine_node({
            "search_results": search_results,
            "intent": intent
        })
        
        return {
            "combined_results": combined_result.get("combined_results", search_results),
            "final_context": combined_result.get("final_context", "")
        }


class AnswerNode(BaseNodeHandler):
    """Handles final answer generation with streaming support."""
    
    def __init__(self):
        super().__init__("answer")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute answer generation logic."""
        normalized_query = state.get("normalized_query", "")
        final_context = state.get("final_context", "")
        combined_results = state.get("combined_results", [])
        intent = state.get("intent")
        
        # Generate streaming response - collect all chunks
        response_chunks = []
        async for chunk in generate_response(
            query=normalized_query,
            context=final_context,
            intent=intent,
            chat_client=None,  # Will be injected by resource manager
            chat_history=[],
            model_name="gpt-3.5-turbo",
            temperature=0.2
        ):
            response_chunks.append(chunk)
        
        final_answer = "".join(response_chunks)
        
        # Verify answer quality
        answer_verification = verify_answer(final_answer, final_context, normalized_query)
        
        # Extract source chips for UI
        source_chips = extract_source_chips(combined_results, max_chips=5)
        
        return {
            "final_answer": final_answer,
            "response_chunks": response_chunks,
            "answer_verification": answer_verification,
            "source_chips": [chip.dict() for chip in source_chips]
        }


class RestartNode(BaseNodeHandler):
    """Handles restart/reset requests."""
    
    def __init__(self):
        super().__init__("restart")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute restart logic - clear context and provide fresh start message."""
        return {
            "final_answer": "Context cleared. I'm ready to help with your next question!",
            "response_chunks": ["Context cleared. I'm ready to help with your next question!"],
            "search_results": [],
            "combined_results": [],
            "final_context": "",
            "error_messages": []
        }


class ListHandlerNode(BaseNodeHandler):
    """Handles list queries using OpenSearch aggregations."""
    
    def __init__(self):
        super().__init__("list_handler")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list generation logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Extract list type from query
        list_type = self._extract_list_type(normalized_query)
        
        # Get list from OpenSearch aggregations
        # This is a placeholder - implement actual aggregation logic
        items = await self._get_list_from_opensearch(list_type)
        
        # Format response
        formatted_response = self._format_list_response(items, list_type, normalized_query)
        
        return {
            "final_answer": formatted_response,
            "response_chunks": [formatted_response],
            "search_results": [],  # List queries don't need search results
            "list_items": items
        }
    
    def _extract_list_type(self, query: str) -> str:
        """Extract what type of list is being requested."""
        query_lower = query.lower()
        
        if "api" in query_lower:
            return "apis"
        elif "utility" in query_lower or "service" in query_lower:
            return "utilities"
        elif "field" in query_lower or "parameter" in query_lower:
            return "fields"
        else:
            return "items"
    
    async def _get_list_from_opensearch(self, list_type: str) -> List[str]:
        """
        Get list items using OpenSearch aggregations.
        
        This is a placeholder - implement actual OpenSearch aggregation logic.
        """
        # Placeholder implementation
        if list_type == "apis":
            return ["Customer-Summary-API", "Account-Balance-API", "Transaction-History-API"]
        elif list_type == "utilities":
            return ["Customer Summary Utility", "Account Utility", "Transaction Utility"]
        else:
            return ["Item 1", "Item 2", "Item 3"]
    
    def _format_list_response(self, items: List[str], list_type: str, query: str) -> str:
        """Format list items into a readable response."""
        if not items:
            return f"No {list_type} found matching your query."
        
        formatted_items = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        
        return f"""I have knowledge of the following {list_type}:

{formatted_items}

Which specific {list_type[:-1]} would you like to know more about?"""


class WorkflowSynthesizerNode(BaseNodeHandler):
    """Handles workflow/procedure synthesis queries."""
    
    def __init__(self):
        super().__init__("workflow_synthesizer")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow synthesis logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Search for workflow-related content
        # This is a placeholder - implement actual workflow search and synthesis
        workflow_steps = await self._synthesize_workflow(normalized_query)
        
        formatted_response = self._format_workflow_response(workflow_steps)
        
        return {
            "final_answer": formatted_response,
            "response_chunks": [formatted_response],
            "workflow_steps": workflow_steps
        }
    
    async def _synthesize_workflow(self, query: str) -> List[str]:
        """
        Synthesize workflow steps from multiple sources.
        
        This is a placeholder - implement actual workflow synthesis logic.
        """
        return [
            "First, authenticate with your credentials",
            "Next, configure the API endpoint",
            "Then, send your request with proper headers",
            "Finally, handle the response appropriately"
        ]
    
    def _format_workflow_response(self, steps: List[str]) -> str:
        """Format workflow steps into a readable response."""
        if not steps:
            return "No workflow steps found for your query."
        
        formatted_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        
        return f"""Here's the step-by-step workflow:

{formatted_steps}

Would you like more details on any specific step?"""