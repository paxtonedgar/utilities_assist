"""LangGraph agents for the multi-step retrieval workflow."""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import Send, Command
from pydantic import BaseModel, Field

from workflows.state import WorkflowState, WorkflowConfig, add_workflow_step, log_error, update_metrics
from services.models import IntentResult, SearchResult, RetrievalResult
from services.normalize import normalize_query
from services.intent import determine_intent
from services.retrieve import enhanced_rrf_search, bm25_search, knn_search
from embedding_creation import create_single_embedding, EmbeddingError
from infra.resource_manager import RAGResources

logger = logging.getLogger(__name__)

# Pydantic models for structured LLM outputs

class QueryComplexityAnalysis(BaseModel):
    """Analysis of query complexity and decomposition needs."""
    complexity: str = Field(description="Query complexity: simple, complex, multi_part, or comparative")
    decomposition_needed: bool = Field(description="Whether query should be decomposed into sub-queries")
    query_type: str = Field(description="Type: factual, comparative, multi_step, list, or exploratory")
    sub_queries: List[str] = Field(default=[], description="List of sub-queries if decomposition is needed")
    entities: List[str] = Field(default=[], description="Key entities mentioned in the query")
    reasoning: str = Field(description="Explanation of the analysis")


class SearchPlan(BaseModel):
    """Plan for executing search operations."""
    search_tasks: List[Dict[str, Any]] = Field(description="List of search operations to perform")
    parallel_execution: bool = Field(description="Whether searches can be run in parallel")
    priority_order: List[str] = Field(description="Order of search priorities")


# Agent implementations

async def query_decomposer_agent(state: WorkflowState, resources: RAGResources) -> Dict[str, Any]:
    """
    Agent that analyzes query complexity and determines if decomposition is needed.
    
    This agent uses an LLM to understand the query structure and decide on the
    appropriate processing strategy.
    """
    start_time = time.perf_counter()
    
    try:
        query = state["original_query"]
        
        # Normalize the query first
        normalized = normalize_query(query)
        
        # Determine basic intent
        utilities_list = [
            "Customer Summary Utility",
            "Enhanced Transaction Utility", 
            "Account Utility",
            "Customer Interaction Utility",
            "Digital Events",
            "Product Catalog Utility",
            "Global Customer Platform"
        ]
        
        intent = await determine_intent(normalized, resources.chat_client, utilities_list, resources.settings.chat.model)
        
        # Use LLM to analyze query complexity with structured output
        complexity_analyzer = resources.chat_client.with_structured_output(QueryComplexityAnalysis)
        
        analysis_prompt = f"""
        Analyze this user query for complexity and determine if it needs decomposition:
        
        Query: "{query}"
        Normalized: "{normalized}"
        Intent: {intent.intent} (confidence: {intent.confidence})
        
        Consider these factors:
        1. Does the query ask for multiple pieces of information?
        2. Does it require comparison between different entities?
        3. Does it need iterative exploration (answer depends on initial results)?
        4. Are there multiple steps needed to fully answer?
        
        Examples of complex queries:
        - "Compare X and Y" (comparative)
        - "What are the options and which is best?" (multi_step)
        - "List all X and their Y properties" (complex list)
        
        Examples of simple queries:
        - "What is X?" (factual)
        - "How do I do X?" (procedural)
        - "Show me documentation for X" (retrieval)
        """
        
        complexity_analysis = await complexity_analyzer.ainvoke([
            SystemMessage(content="You are an expert at analyzing query complexity for information retrieval systems."),
            HumanMessage(content=analysis_prompt)
        ])
        
        # Calculate metrics
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "normalized_query": normalized,
            "intent": intent,
            "query_complexity": complexity_analysis.complexity,
            "sub_queries": complexity_analysis.sub_queries if complexity_analysis.decomposition_needed else None,
            **add_workflow_step(state, "query_decomposer"),
            **update_metrics(state, {
                "query_analysis_ms": processing_time,
                "complexity_score": intent.confidence
            })
        }
        
    except Exception as e:
        error_msg = f"Query decomposer failed: {e}"
        logger.error(error_msg)
        return {
            **add_workflow_step(state, "query_decomposer"),
            **log_error(state, error_msg)
        }


async def search_orchestrator_agent(state: WorkflowState, resources: RAGResources) -> Command:
    """
    Agent that coordinates multiple search operations based on query complexity.
    
    For simple queries, routes to single search.
    For complex queries, spawns parallel search agents using LangGraph's Send API.
    """
    start_time = time.perf_counter()
    
    try:
        complexity = state["query_complexity"]
        query = state["normalized_query"] or state["original_query"]
        intent = state["intent"]
        
        # Decide on search strategy based on complexity
        if complexity == "simple":
            # Route to single search agent
            return Command(goto="single_search_agent")
            
        elif complexity in ["complex", "multi_part", "comparative"]:
            # Plan parallel searches
            sub_queries = state.get("sub_queries", [query])  # Fallback to main query if no decomposition
            
            # Create search tasks for parallel execution
            search_tasks = []
            
            if complexity == "comparative" and len(sub_queries) >= 2:
                # For comparative queries, search for each entity separately
                for i, sub_query in enumerate(sub_queries[:3]):  # Limit to 3 parallel searches
                    search_tasks.append(Send("parallel_search_agent", {
                        **state,
                        "current_query": sub_query,
                        "search_id": f"search_{i}",
                        "search_context": f"comparative_part_{i}"
                    }))
            
            elif complexity == "multi_part":
                # For multi-part queries, try different search strategies
                search_tasks.extend([
                    Send("parallel_search_agent", {
                        **state,
                        "current_query": query,
                        "search_id": "comprehensive",
                        "search_strategy": "enhanced_rrf"
                    }),
                    Send("parallel_search_agent", {
                        **state,
                        "current_query": query,
                        "search_id": "broad",
                        "search_strategy": "bm25_broad"
                    })
                ])
                
                # Add sub-query searches if available
                for i, sub_query in enumerate(sub_queries[:2]):
                    search_tasks.append(Send("parallel_search_agent", {
                        **state,
                        "current_query": sub_query,
                        "search_id": f"sub_{i}",
                        "search_context": f"sub_query_{i}"
                    }))
            
            else:  # complex
                # Use enhanced search with broader parameters
                search_tasks.append(Send("parallel_search_agent", {
                    **state,
                    "current_query": query,
                    "search_id": "enhanced",
                    "search_strategy": "enhanced_rrf_broad"
                }))
            
            # Update state and dispatch parallel searches
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return Command(
                goto=search_tasks,
                update={
                    **add_workflow_step(state, "search_orchestrator"),
                    **update_metrics(state, {
                        "orchestration_ms": processing_time,
                        "parallel_searches": len(search_tasks)
                    })
                }
            )
            
    except Exception as e:
        error_msg = f"Search orchestrator failed: {e}"
        logger.error(error_msg)
        return Command(
            goto="error_handler",
            update={
                **add_workflow_step(state, "search_orchestrator"),
                **log_error(state, error_msg)
            }
        )


async def single_search_agent(state: WorkflowState, resources: RAGResources) -> Dict[str, Any]:
    """
    Agent that performs a single, optimized search for simple queries.
    
    Uses the existing enhanced RRF search for high-quality results.
    """
    start_time = time.perf_counter()
    
    try:
        query = state["normalized_query"] or state["original_query"]
        intent = state["intent"]
        
        # Determine search parameters
        use_mock_corpus = state.get("user_context", {}).get("use_mock_corpus", False)
        index_name = "confluence_mock" if use_mock_corpus else resources.settings.search.index_alias
        
        # Use enhanced search if embedding client is available and confidence is high
        if resources.embed_client and intent.confidence > 0.7:
            try:
                # Get embedding
                expected_dims = 1536
                query_embedding = await create_single_embedding(
                    embed_client=resources.embed_client,
                    text=query,
                    model=resources.settings.embed.model,
                    expected_dims=expected_dims
                )
                
                # Enhanced RRF search
                result, diagnostics = await enhanced_rrf_search(
                    query=query,
                    query_embedding=query_embedding,
                    search_client=resources.search_client,
                    index_name=index_name,
                    top_k=10,
                    use_mmr=True,
                    lambda_param=0.75
                )
                
            except EmbeddingError:
                # Fallback to BM25
                result = await bm25_search(
                    query=query,
                    search_client=resources.search_client,
                    index_name=index_name,
                    top_k=10
                )
        else:
            # Use BM25 for low confidence or no embedding client
            result = await bm25_search(
                query=query,
                search_client=resources.search_client,
                index_name=index_name,
                top_k=10
            )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "search_results": result.results if result else [],
            "result_sources": {"single_search": result.results if result else []},
            **add_workflow_step(state, "single_search"),
            **update_metrics(state, {
                "single_search_ms": processing_time,
                "results_found": len(result.results) if result else 0
            })
        }
        
    except Exception as e:
        error_msg = f"Single search agent failed: {e}"
        logger.error(error_msg)
        return {
            **add_workflow_step(state, "single_search"),
            **log_error(state, error_msg)
        }


async def parallel_search_agent(state: WorkflowState, resources: RAGResources) -> Dict[str, Any]:
    """
    Agent that performs individual search operations in parallel workflows.
    
    This agent is spawned multiple times by the search orchestrator for complex queries.
    """
    start_time = time.perf_counter()
    
    try:
        query = state.get("current_query", state["normalized_query"] or state["original_query"])
        search_id = state.get("search_id", "parallel")
        search_strategy = state.get("search_strategy", "enhanced_rrf")
        search_context = state.get("search_context", "")
        
        # Determine search parameters
        use_mock_corpus = state.get("user_context", {}).get("use_mock_corpus", False)
        index_name = "confluence_mock" if use_mock_corpus else resources.settings.search.index_alias
        
        # Adjust search parameters based on strategy
        if search_strategy == "enhanced_rrf_broad":
            top_k = 15
            lambda_param = 0.6  # More diverse results
        elif search_strategy == "bm25_broad":
            top_k = 20
        else:
            top_k = 10
            lambda_param = 0.75
        
        # Execute search based on strategy
        result = None
        
        if search_strategy.startswith("enhanced_rrf") and resources.embed_client:
            try:
                # Get embedding
                expected_dims = 1536
                query_embedding = await create_single_embedding(
                    embed_client=resources.embed_client,
                    text=query,
                    model=resources.settings.embed.model,
                    expected_dims=expected_dims
                )
                
                # Enhanced RRF search
                result, diagnostics = await enhanced_rrf_search(
                    query=query,
                    query_embedding=query_embedding,
                    search_client=resources.search_client,
                    index_name=index_name,
                    top_k=top_k,
                    use_mmr=True,
                    lambda_param=lambda_param
                )
                
            except EmbeddingError:
                # Fallback to BM25
                result = await bm25_search(
                    query=query,
                    search_client=resources.search_client,
                    index_name=index_name,
                    top_k=top_k
                )
        else:
            # BM25 search
            result = await bm25_search(
                query=query,
                search_client=resources.search_client,
                index_name=index_name,
                top_k=top_k
            )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Tag results with search context for later synthesis
        if result and result.results:
            for search_result in result.results:
                search_result.metadata[f"search_id"] = search_id
                search_result.metadata[f"search_context"] = search_context
                search_result.metadata[f"source_query"] = query
        
        return {
            "search_results": result.results if result else [],
            **add_workflow_step(state, f"parallel_search_{search_id}"),
            **update_metrics(state, {
                f"parallel_search_{search_id}_ms": processing_time,
                f"results_{search_id}": len(result.results) if result else 0
            })
        }
        
    except Exception as e:
        error_msg = f"Parallel search agent {state.get('search_id', 'unknown')} failed: {e}"
        logger.error(error_msg)
        return {
            **add_workflow_step(state, f"parallel_search_{state.get('search_id', 'error')}"),
            **log_error(state, error_msg)
        }


async def result_synthesizer_agent(state: WorkflowState, resources: RAGResources) -> Dict[str, Any]:
    """
    Agent that intelligently combines and synthesizes results from multiple searches.
    
    Uses LLM-based analysis to create coherent context from diverse search results.
    """
    start_time = time.perf_counter()
    
    try:
        all_results = state["search_results"]
        query = state["normalized_query"] or state["original_query"] 
        complexity = state["query_complexity"]
        
        if not all_results:
            return {
                "synthesized_context": "No relevant information found.",
                **add_workflow_step(state, "result_synthesizer")
            }
        
        # Group results by search context for intelligent synthesis
        grouped_results = {}
        for result in all_results:
            search_context = result.metadata.get("search_context", "general")
            if search_context not in grouped_results:
                grouped_results[search_context] = []
            grouped_results[search_context].append(result)
        
        # Create synthesis strategy based on query complexity
        if complexity == "comparative":
            context = await _synthesize_comparative_results(grouped_results, query, resources.chat_client)
        elif complexity == "multi_part":
            context = await _synthesize_multipart_results(all_results, query, resources.chat_client)
        else:
            # Standard synthesis with deduplication and ranking
            context = await _synthesize_standard_results(all_results, query, resources.chat_client)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "synthesized_context": context,
            "result_sources": grouped_results,
            **add_workflow_step(state, "result_synthesizer"),
            **update_metrics(state, {
                "synthesis_ms": processing_time,
                "total_results_processed": len(all_results),
                "grouped_contexts": len(grouped_results)
            })
        }
        
    except Exception as e:
        error_msg = f"Result synthesizer failed: {e}"
        logger.error(error_msg)
        return {
            **add_workflow_step(state, "result_synthesizer"),
            **log_error(state, error_msg)
        }


# Helper functions for different synthesis strategies

async def _synthesize_comparative_results(grouped_results: Dict[str, List[SearchResult]], 
                                        query: str, chat_client) -> str:
    """Synthesize results for comparative queries."""
    if len(grouped_results) < 2:
        # Not enough groups for comparison, use standard synthesis
        all_results = []
        for results in grouped_results.values():
            all_results.extend(results)
        return _build_standard_context(all_results)
    
    # Create comparative context
    comparison_sections = []
    for context_name, results in grouped_results.items():
        if results:
            section = f"\n=== Information about {context_name.replace('_', ' ').title()} ===\n"
            for result in results[:3]:  # Limit to top 3 per section
                section += f"- {result.content[:300]}...\n"
            comparison_sections.append(section)
    
    return "\n".join(comparison_sections)


async def _synthesize_multipart_results(all_results: List[SearchResult], 
                                      query: str, chat_client) -> str:
    """Synthesize results for multi-part queries."""
    # Use LLM to create structured synthesis
    synthesis_prompt = f"""
    Synthesize the following search results to comprehensively answer this multi-part query: "{query}"
    
    Search Results:
    {_format_results_for_llm(all_results[:15])}
    
    Create a structured response that addresses all aspects of the query.
    Organize the information logically and highlight key points.
    """
    
    try:
        response = await chat_client.ainvoke([
            SystemMessage(content="You are an expert at synthesizing information from multiple sources into coherent, comprehensive responses."),
            HumanMessage(content=synthesis_prompt)
        ])
        return response.content
    except Exception as e:
        logger.warning(f"LLM synthesis failed, using standard approach: {e}")
        return _build_standard_context(all_results)


async def _synthesize_standard_results(all_results: List[SearchResult], 
                                     query: str, chat_client) -> str:
    """Standard result synthesis with deduplication."""
    return _build_standard_context(all_results)


def _build_standard_context(results: List[SearchResult], max_length: int = 8000) -> str:
    """Build standard context from search results."""
    if not results:
        return "No relevant information found."
    
    context_parts = []
    current_length = 0
    
    # Sort by relevance score
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    
    for result in sorted_results:
        content = result.content[:500]  # Limit individual result length
        title = result.metadata.get("title", "Document")
        
        part = f"\nFrom: {title}\n{content}\n"
        
        if current_length + len(part) > max_length:
            break
            
        context_parts.append(part)
        current_length += len(part)
    
    return "".join(context_parts)


def _format_results_for_llm(results: List[SearchResult]) -> str:
    """Format results for LLM processing."""
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.metadata.get("title", f"Result {i}")
        content = result.content[:400]  # Limit for LLM context
        formatted.append(f"{i}. {title}\n{content}\n")
    
    return "\n".join(formatted)