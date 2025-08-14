"""
Main LangGraph definition with branching and loops.

This implements the core graph: Summarize → Intent → branch logic with
support for compound questions and iterative refinement.
"""

import logging
from typing import Dict, Any, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.summarize import summarize_node
from agent.nodes.intent import intent_node  
from agent.nodes.combine import combine_node
from agent.tools.search import adaptive_search_tool, multi_index_search_tool
from services.models import SearchResult, IntentResult, RetrievalResult
from services.respond import generate_response
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Load answer template
template_dir = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


class GraphState(Dict[str, Any]):
    """State for the LangGraph workflow with user context and authentication."""
    # Core data
    original_query: str
    normalized_query: Optional[str]
    intent: Optional[IntentResult]
    
    # Search results (accumulated)
    search_results: List[SearchResult]
    combined_results: List[SearchResult] 
    final_context: Optional[str]
    
    # Response
    final_answer: Optional[str]
    response_chunks: List[str]
    
    # User context and authentication (CRITICAL ADDITION)
    user_id: Optional[str]
    thread_id: Optional[str] 
    session_id: Optional[str]
    user_context: Optional[Dict[str, Any]]
    user_preferences: Optional[Dict[str, Any]]
    
    # Control flow
    workflow_path: List[str]
    loop_count: int
    coverage_threshold: float
    min_results: int
    
    # Error handling
    error_messages: List[str]
    
    # Configuration (not clients - follows LangGraph pattern)
    _use_mock_corpus: bool


def create_graph(
    enable_loops: bool = True, 
    coverage_threshold: float = 0.7, 
    min_results: int = 3,
    checkpointer=None,
    store=None
) -> StateGraph:
    """
    Create the main LangGraph with branching, optional loops, and persistence.
    
    Graph structure:
    Summarize → Intent → Branch:
      - if swagger → Search(index="swagger")  
      - if confluence → Search(index="confluence")
      - if workflow/comparative → Multi-search
      - Optional loop: if hits < K or coverage < threshold → rewrite → re-search
    → Combine → Answer
    
    Args:
        enable_loops: Whether to enable iterative refinement loops
        coverage_threshold: Minimum coverage score to avoid re-search
        min_results: Minimum number of results required
        checkpointer: Optional checkpointer for conversation persistence
        store: Optional store for cross-thread user memory
        
    Returns:
        Compiled StateGraph ready for execution with persistence
    """
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("summarize", _summarize_wrapper)
    workflow.add_node("intent", _intent_wrapper)
    workflow.add_node("search_confluence", _search_confluence_wrapper)
    workflow.add_node("search_swagger", _search_swagger_wrapper) 
    workflow.add_node("search_multi", _search_multi_wrapper)
    workflow.add_node("restart", _restart_wrapper)  # MISSING: Restart intent handler
    workflow.add_node("rewrite_query", _rewrite_query_wrapper)
    workflow.add_node("combine", _combine_wrapper)
    workflow.add_node("answer", _answer_wrapper)
    
    # Define the main flow
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "intent")
    
    # Intent-based branching (including restart handling)
    workflow.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "search_confluence": "search_confluence",
            "search_swagger": "search_swagger", 
            "search_multi": "search_multi",
            "restart": "restart"  # MISSING: Handle restart intent
        }
    )
    
    # Optional loop logic (if enabled)
    if enable_loops:
        # After each search, check if we need to loop
        workflow.add_conditional_edges(
            "search_confluence", 
            lambda state: _check_coverage(state, coverage_threshold, min_results),
            {
                "rewrite": "rewrite_query",
                "combine": "combine"
            }
        )
        workflow.add_conditional_edges(
            "search_swagger",
            lambda state: _check_coverage(state, coverage_threshold, min_results), 
            {
                "rewrite": "rewrite_query",
                "combine": "combine"
            }
        )
        workflow.add_conditional_edges(
            "search_multi",
            lambda state: _check_coverage(state, coverage_threshold, min_results),
            {
                "rewrite": "rewrite_query", 
                "combine": "combine"
            }
        )
        
        # After rewrite, determine which search to retry
        workflow.add_conditional_edges(
            "rewrite_query",
            _route_after_rewrite,
            {
                "search_confluence": "search_confluence",
                "search_swagger": "search_swagger",
                "search_multi": "search_multi"
            }
        )
    else:
        # Direct routing to combine without loops
        workflow.add_edge("search_confluence", "combine")
        workflow.add_edge("search_swagger", "combine")
        workflow.add_edge("search_multi", "combine")
    
    # Final steps
    workflow.add_edge("combine", "answer")
    workflow.add_edge("answer", END)
    workflow.add_edge("restart", END)  # MISSING: Direct restart to end
    
    # Compile with optional persistence components
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
        logger.info("Graph compiled with checkpointer for conversation persistence")
    if store is not None:
        compile_kwargs["store"] = store  
        logger.info("Graph compiled with store for cross-thread user memory")
    
    return workflow.compile(**compile_kwargs)


def _route_after_intent(state: GraphState) -> Literal["search_confluence", "search_swagger", "search_multi", "restart"]:
    """Route to appropriate search based on intent classification."""
    intent = state.get("intent")
    if not intent:
        return "search_confluence"  # Default fallback
    
    intent_type = intent.intent.lower()
    
    # Handle restart intent specially (MISSING from original implementation)
    if intent_type == "restart":
        return "restart"
    
    # Check for compound/comparative queries that need multi-search
    query = state.get("normalized_query", "").lower()
    comparative_keywords = ["compare", "versus", "vs", "difference", "which is better"]
    multi_entity_indicators = ["and", "both", "either", "all"]
    
    is_comparative = any(keyword in query for keyword in comparative_keywords)
    has_multiple_entities = any(indicator in query for indicator in multi_entity_indicators)
    
    if is_comparative or has_multiple_entities or intent_type in ["workflow", "comparative"]:
        return "search_multi"
    elif intent_type == "swagger":
        return "search_swagger"  
    else:
        return "search_confluence"


def _route_after_rewrite(state: GraphState) -> Literal["search_confluence", "search_swagger", "search_multi"]:
    """Route after query rewrite - use same logic as initial routing."""
    return _route_after_intent(state)


def _check_coverage(state: GraphState, coverage_threshold: float, min_results: int) -> Literal["rewrite", "combine"]:
    """
    Check if we have sufficient coverage to proceed or need to rewrite query.
    
    This implements the loop condition: if hits < K or coverage < threshold → rewrite.
    """
    results = state.get("search_results", [])
    loop_count = state.get("loop_count", 0)
    
    # Prevent infinite loops
    if loop_count >= 2:
        logger.info("Max loop count reached, proceeding to combine")
        return "combine"
    
    # Check result count
    if len(results) < min_results:
        logger.info(f"Insufficient results ({len(results)} < {min_results}), rewriting query")
        return "rewrite"
    
    # Check coverage score (simplified - could use more sophisticated metrics)
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        if avg_score < coverage_threshold:
            logger.info(f"Low coverage score ({avg_score:.2f} < {coverage_threshold}), rewriting query")
            return "rewrite"
    
    # Sufficient coverage, proceed
    return "combine"


# Node wrapper functions that handle resources and state management

async def _summarize_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Wrapper for summarize node - follows LangGraph pattern."""
    result = await summarize_node(state, config, store=store)
    return result


async def _intent_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Wrapper for intent node - follows LangGraph pattern."""
    result = await intent_node(state, config, store=store)
    return result


async def _search_confluence_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Search confluence index - follows LangGraph pattern."""
    try:
        # Extract resources from global resource manager
        from infra.resource_manager import get_resources
        resources = get_resources()
        
        query = state["normalized_query"]
        intent = state.get("intent")
        use_mock = state.get("_use_mock_corpus", False)
        
        # Use the configured search index from resources instead of hardcoding
        index = "confluence_mock" if use_mock else resources.settings.search.index_alias
        
        result = await adaptive_search_tool(
            query=query,
            intent_confidence=intent.confidence if intent else 0.5,
            intent_type="confluence",
            search_client=resources.search_client,
            embed_client=resources.embed_client,
            embed_model=resources.settings.embed.model,
            use_mock_corpus=use_mock,
            top_k=10
        )
        
        # Mark results with search method
        for search_result in result.results:
            search_result.metadata["search_method"] = "confluence"
            search_result.metadata["search_id"] = "confluence"
        
        return {
            "search_results": result.results,
            "workflow_path": state.get("workflow_path", []) + ["search_confluence"]
        }
        
    except Exception as e:
        logger.error(f"Confluence search failed: {e}")
        return {
            "search_results": [],
            "workflow_path": state.get("workflow_path", []) + ["search_confluence_error"],
            "error_messages": state.get("error_messages", []) + [f"Confluence search failed: {e}"]
        }


async def _search_swagger_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Search swagger index - follows LangGraph pattern.""" 
    try:
        # Extract resources from global resource manager
        from infra.resource_manager import get_resources
        resources = get_resources()
        
        query = state["normalized_query"]
        intent = state.get("intent")
        
        result = await adaptive_search_tool(
            query=query,
            intent_confidence=intent.confidence if intent else 0.5,
            intent_type="swagger",
            search_client=resources.search_client,
            embed_client=resources.embed_client,
            embed_model=resources.settings.embed.model,
            use_mock_corpus=False,  # Swagger is never mocked
            top_k=10
        )
        
        # Mark results with search method
        for search_result in result.results:
            search_result.metadata["search_method"] = "swagger"
            search_result.metadata["search_id"] = "swagger"
        
        return {
            "search_results": result.results,
            "workflow_path": state.get("workflow_path", []) + ["search_swagger"]
        }
        
    except Exception as e:
        logger.error(f"Swagger search failed: {e}")
        return {
            "search_results": [],
            "workflow_path": state.get("workflow_path", []) + ["search_swagger_error"],
            "error_messages": state.get("error_messages", []) + [f"Swagger search failed: {e}"]
        }


async def _search_multi_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """
    Multi-index search for compound questions - follows LangGraph pattern.
    
    For compound queries, search multiple indices and combine results.
    This enables comparison queries and comprehensive coverage.
    """
    try:
        # Extract resources from global resource manager
        from infra.resource_manager import get_resources
        resources = get_resources()
        
        query = state["normalized_query"]
        intent = state.get("intent")
        use_mock = state.get("_use_mock_corpus", False)
        
        # Define indices to search for compound queries
        indices = []
        if use_mock:
            indices.append("confluence_mock")
        else:
            indices.append(resources.settings.search.index_alias)  # Use configured index
            indices.append("khub-opensearch-swagger-index")
        
        # Search all indices
        results_list = await multi_index_search_tool(
            indices=indices,
            query=query,
            search_client=resources.search_client,
            embed_client=resources.embed_client,
            embed_model=resources.settings.embed.model,
            top_k_per_index=8  # Get fewer per index to avoid overwhelming context
        )
        
        # Combine results from all indices
        all_results = []
        for i, result in enumerate(results_list):
            index_name = indices[i] if i < len(indices) else f"index_{i}"
            for search_result in result.results:
                search_result.metadata["search_method"] = "multi_index"
                search_result.metadata["search_id"] = f"multi_{index_name}"
                all_results.append(search_result)
        
        logger.info(f"Multi-search found {len(all_results)} results across {len(indices)} indices")
        
        return {
            "search_results": all_results,
            "workflow_path": state.get("workflow_path", []) + ["search_multi"]
        }
        
    except Exception as e:
        logger.error(f"Multi-index search failed: {e}")
        return {
            "search_results": [],
            "workflow_path": state.get("workflow_path", []) + ["search_multi_error"],
            "error_messages": state.get("error_messages", []) + [f"Multi-search failed: {e}"]
        }


async def _rewrite_query_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """
    Rewrite query for better search results - follows LangGraph pattern.
    
    This implements the loop logic: if coverage is insufficient, 
    use LLM to rewrite the query and try again.
    """
    try:
        # Extract resources from global resource manager
        from infra.resource_manager import get_resources
        resources = get_resources()
        
        original_query = state["normalized_query"]
        current_results = state.get("search_results", [])
        loop_count = state.get("loop_count", 0)
        
        # Analyze current results to understand what's missing
        result_titles = [r.metadata.get("title", "") for r in current_results[:5]]
        result_summary = "; ".join(result_titles[:3])
        
        rewrite_prompt = f"""
        The original query "{original_query}" returned {len(current_results)} results, but coverage seems insufficient.
        
        Current results include: {result_summary}
        
        Please rewrite this query to find more comprehensive information. Consider:
        - Using different keywords or synonyms
        - Being more specific about what's needed
        - Expanding the scope if the query was too narrow
        - Focusing on key concepts if the query was too broad
        
        Rewritten query:
        """
        
        # Create LangChain wrapper for Azure OpenAI client using exact config.ini field names
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Get JPMC-specific headers from the original client and filter out problematic ones
        raw_headers = getattr(resources.chat_client, 'default_headers', {})
        default_headers = {}
        for key, value in raw_headers.items():
            # Filter out openai.Omit objects that LangChain can't handle
            if hasattr(value, '__class__') and 'Omit' in str(type(value)):
                continue
            if isinstance(value, str):
                default_headers[key] = value
        
        langchain_client = AzureChatOpenAI(
            api_version=resources.settings.chat.api_version,  # api_version from config
            azure_deployment=resources.settings.chat.model,   # deployment_name from config  
            azure_endpoint=resources.settings.chat.api_base,  # azure_openai_endpoint from config
            api_key=resources.chat_client.api_key,            # api_key from config
            default_headers=default_headers,                  # JPMC headers (user_sid, etc.)
            temperature=0.1,
            max_tokens=200
        )
        
        response = await langchain_client.ainvoke([
            SystemMessage(content="You are a query optimization expert. Rewrite queries to improve search results."),
            HumanMessage(content=rewrite_prompt)
        ])
        
        rewritten_query = response.content.strip()
        logger.info(f"Query rewritten: '{original_query}' -> '{rewritten_query}'")
        
        return {
            "normalized_query": rewritten_query,
            "loop_count": loop_count + 1,
            "search_results": [],  # Clear previous results for fresh search
            "workflow_path": state.get("workflow_path", []) + ["rewrite_query"]
        }
        
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return {
            "loop_count": state.get("loop_count", 0) + 1,  # Still increment to prevent infinite loops
            "workflow_path": state.get("workflow_path", []) + ["rewrite_query_error"],
            "error_messages": state.get("error_messages", []) + [f"Query rewrite failed: {e}"]
        }


async def _combine_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Wrapper for combine node - follows LangGraph pattern."""
    result = await combine_node(state, config, store=store)
    return result


async def _restart_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Handle restart intent - MISSING from original LangGraph implementation."""
    restart_message = "Context has been cleared. How can I help you today?"
    
    return {
        "final_answer": restart_message,
        "response_chunks": [restart_message],
        "combined_results": [],
        "workflow_path": state.get("workflow_path", []) + ["restart"]
    }


async def _answer_wrapper(state: GraphState, *, config=None, store=None) -> Dict[str, Any]:
    """Generate final answer using jinja template with verification - follows LangGraph pattern."""
    try:
        # Extract resources from global resource manager
        from infra.resource_manager import get_resources
        resources = get_resources()
        
        query = state.get("normalized_query", state.get("original_query", ""))
        intent = state.get("intent")
        context = state.get("final_context", "")
        
        if not context or context == "No relevant information found.":
            return {
                "final_answer": "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic.",
                "response_chunks": ["I couldn't find relevant information to answer your question."],
                "workflow_path": state.get("workflow_path", []) + ["answer"]
            }
        
        # Use jinja template for answer generation
        try:
            template = jinja_env.get_template("answer.jinja")
            prompt = template.render(
                query=query,
                intent=intent or IntentResult(intent="unknown", confidence=0.0),
                context=context,
                chat_history=[]  # Could be expanded to include from state
            )
            
            # Generate streaming response
            response_chunks = []
            full_answer = ""
            
            # Get LLM parameters
            temperature = float(resources.get_config_param('temperature', 0.2))
            max_tokens = int(resources.get_config_param('max_tokens_2k', 1500))
            
            async for chunk in generate_response(
                query,
                context, 
                intent,
                resources.chat_client,
                [],  # chat_history
                resources.settings.chat.model,
                temperature,
                max_tokens,
                system_prompt_override=prompt
            ):
                full_answer += chunk
                response_chunks.append(chunk)
            
        except Exception as e:
            logger.warning(f"Template-based answer generation failed, using fallback: {e}")
            # Fallback to simple context-based answer
            full_answer = f"Based on the available information:\n\n{context[:1000]}..."
            response_chunks = [full_answer]
        
        # MISSING: Answer verification (from traditional pipeline)
        verification_metrics = _verify_answer_quality(full_answer, context, query)
        logger.info(f"Answer verification: confidence={verification_metrics.get('confidence_score', 0.0):.2f}")
        
        # MISSING: Extract source chips (from traditional pipeline)
        combined_results = state.get("combined_results", [])
        source_chips = _extract_source_chips(combined_results)
        
        return {
            "final_answer": full_answer,
            "response_chunks": response_chunks,
            "source_chips": source_chips,  # MISSING: Source chips for UI
            "verification_metrics": verification_metrics,  # MISSING: Quality metrics
            "workflow_path": state.get("workflow_path", []) + ["answer"]
        }
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            "final_answer": f"I encountered an error while generating the response: {str(e)}",
            "response_chunks": [f"Error: {str(e)}"],
            "workflow_path": state.get("workflow_path", []) + ["answer_error"],
            "error_messages": state.get("error_messages", []) + [f"Answer generation failed: {e}"]
        }


def _verify_answer_quality(answer: str, context: str, query: str) -> Dict[str, Any]:
    """MISSING FUNCTION: Answer verification from traditional pipeline."""
    try:
        from services.respond import verify_answer
        return verify_answer(answer, context, query)
    except ImportError:
        # Minimal verification if respond service not available
        return {
            "has_content": len(answer.strip()) > 10,
            "confidence_score": 0.8 if len(answer.strip()) > 50 else 0.3
        }


def _extract_source_chips(results: List[SearchResult], max_chips: int = 5) -> List[Dict[str, Any]]:
    """MISSING FUNCTION: Source chips extraction from traditional pipeline."""
    try:
        from services.respond import extract_source_chips
        source_chips = extract_source_chips(results, max_chips)
        return [chip.dict() if hasattr(chip, 'dict') else chip for chip in source_chips]
    except ImportError:
        # Basic chip extraction if respond service not available
        chips = []
        for result in results[:max_chips]:
            chips.append({
                "title": result.metadata.get("title", "Document"),
                "doc_id": result.doc_id,
                "url": result.metadata.get("url", "#"),
                "score": result.score
            })
        return chips


# Node registry for external access
NODE_REGISTRY = {
    "summarize": _summarize_wrapper,
    "intent": _intent_wrapper,
    "search_confluence": _search_confluence_wrapper,
    "search_swagger": _search_swagger_wrapper,
    "search_multi": _search_multi_wrapper,
    "restart": _restart_wrapper,  # MISSING: Restart handler
    "rewrite_query": _rewrite_query_wrapper,
    "combine": _combine_wrapper,
    "answer": _answer_wrapper
}