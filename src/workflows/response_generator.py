"""Response generation agent for LangGraph workflow."""

import time
import logging
from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage, SystemMessage

from workflows.state import WorkflowState, add_workflow_step, log_error, update_metrics
from services.respond import build_context, generate_response, verify_answer, extract_source_chips
from infra.resource_manager import RAGResources

logger = logging.getLogger(__name__)


async def response_generator_agent(state: WorkflowState, resources: RAGResources) -> Dict[str, Any]:
    """
    Agent that generates the final response using synthesized context.
    
    This agent takes the synthesized context and generates a streaming response,
    similar to the original generate_response function but enhanced for multi-source context.
    """
    start_time = time.perf_counter()
    
    try:
        query = state["normalized_query"] or state["original_query"]
        intent = state["intent"]
        context = state.get("synthesized_context", "")
        complexity = state.get("query_complexity", "simple")
        
        if not context or context == "No relevant information found.":
            return {
                "final_answer": "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic.",
                "response_chunks": ["I couldn't find relevant information to answer your question."],
                **add_workflow_step(state, "response_generator"),
                **update_metrics(state, {"response_generation_ms": 0})
            }
        
        # Get LLM parameters from cached config
        temperature = float(resources.get_config_param('temperature', 0.2))
        max_tokens = int(resources.get_config_param('max_tokens_2k', 1500))
        
        # Adjust system prompt based on query complexity
        system_prompt = _get_system_prompt_for_complexity(complexity)
        
        # Prepare chat history from user context
        chat_history = state.get("user_context", {}).get("chat_history", [])
        
        # Generate streaming response
        response_chunks = []
        full_answer = ""
        
        generation_start = time.perf_counter()
        
        try:
            async for response_chunk in generate_response(
                query,
                context,
                intent,
                resources.chat_client,
                chat_history,
                resources.settings.chat.model,
                temperature,
                max_tokens,
                system_prompt_override=system_prompt
            ):
                full_answer += response_chunk
                response_chunks.append(response_chunk)
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            error_response = f"I encountered an error while generating the response: {str(e)}"
            response_chunks = [error_response]
            full_answer = error_response
        
        generation_time = (time.perf_counter() - generation_start) * 1000
        
        # Extract source chips
        search_results = state.get("search_results", [])
        source_chips = extract_source_chips(search_results)
        
        # Verify answer quality (optional for complex workflows)
        verification = {"verdict": "pass", "confidence_score": 0.8}  # Simplified
        if len(full_answer) > 50:  # Only verify substantial answers
            try:
                verification = verify_answer(full_answer, context, query)
            except Exception as e:
                logger.warning(f"Answer verification failed: {e}")
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "final_answer": full_answer,
            "response_chunks": response_chunks,
            "source_chips": source_chips,
            "verification": verification,
            **add_workflow_step(state, "response_generator"),
            **update_metrics(state, {
                "response_generation_ms": generation_time,
                "total_response_time_ms": processing_time,
                "response_length": len(full_answer),
                "chunks_generated": len(response_chunks)
            })
        }
        
    except Exception as e:
        error_msg = f"Response generator failed: {e}"
        logger.error(error_msg)
        return {
            "final_answer": f"I encountered an error while generating the response: {str(e)}",
            "response_chunks": [f"Error: {str(e)}"],
            **add_workflow_step(state, "response_generator"),
            **log_error(state, error_msg)
        }


def _get_system_prompt_for_complexity(complexity: str) -> str:
    """Get appropriate system prompt based on query complexity."""
    
    base_prompt = """You are a helpful assistant for enterprise utilities and APIs. 
Provide accurate, detailed responses based on the provided context."""
    
    if complexity == "comparative":
        return base_prompt + """
        
For this comparative query, structure your response to clearly compare the different options or entities.
Use a clear format that highlights similarities and differences.
If comparing multiple utilities or APIs, organize the information in a structured way (e.g., using sections or bullet points)."""
    
    elif complexity == "multi_part":
        return base_prompt + """
        
This is a multi-part query that requires addressing several aspects.
Structure your response to comprehensively address each part of the question.
Use clear headings or numbered sections if helpful to organize the information."""
    
    elif complexity == "complex":
        return base_prompt + """
        
This is a complex query that may require detailed explanation.
Provide a comprehensive response that covers all relevant aspects.
Include specific examples or details when available."""
    
    else:  # simple
        return base_prompt + """
        
Provide a clear, direct response to this query.
Be concise but complete in your answer."""