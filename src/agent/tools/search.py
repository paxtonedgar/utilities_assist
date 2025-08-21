"""Search tools for LangGraph workflow - wrapping existing search functions."""

import logging
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

from src.services.retrieve import enhanced_rrf_search, bm25_search, knn_search
from src.embedding_creation import create_single_embedding, EmbeddingError
from src.infra.opensearch_client import OpenSearchClient
from src.infra.search_config import OpenSearchConfig
from src.services.models import RetrievalResult
from src.telemetry.logger import stage

logger = logging.getLogger(__name__)


async def _create_enhanced_embedding(query: str, embed_client, embed_model: str) -> List[float]:
    """Create enhanced embedding with acronym expansion and domain context."""
    try:
        from agent.acronym_map import expand_acronym, get_apis_for_acronym
        expanded_query, expansions = expand_acronym(query)
        
        # Build rich embedding text: "{user_query} ({expanded_full_name}) {api_keywords_joined} site:Utilities"
        embedding_parts = [query]
        
        if expansions:
            # Add expanded name in parentheses
            embedding_parts.append(f"({expansions[0]})")
            
            # Add associated API names as keywords
            acronym = query.upper().split()[0]
            api_names = get_apis_for_acronym(acronym)
            if api_names:
                # Extract key terms from API names for embedding
                api_keywords = []
                for api_name in api_names[:3]:  # Limit to top 3 APIs
                    # Extract meaningful keywords from API names
                    words = api_name.replace("API", "").replace("Service", "").split()
                    api_keywords.extend([w for w in words if len(w) > 2])
                
                if api_keywords:
                    embedding_parts.append(" ".join(api_keywords[:5]))  # Top 5 keywords
        
        # Add domain and context hints
        embedding_parts.append("site:Utilities onboarding runbook documentation")
        
        embedding_query = " ".join(embedding_parts)
        
        expected_dims = OpenSearchConfig.EMBEDDING_DIMENSIONS
        return await create_single_embedding(
            embed_client=embed_client,
            text=embedding_query,
            model=embed_model,
            expected_dims=expected_dims
        )
    except Exception as e:
        logger.warning(f"Enhanced embedding failed, using simple query: {e}")
        # Fallback to simple query embedding
        expected_dims = OpenSearchConfig.EMBEDDING_DIMENSIONS
        return await create_single_embedding(
            embed_client=embed_client,
            text=query,
            model=embed_model,
            expected_dims=expected_dims
        )



@stage("search_execution")
async def search_index_tool(
    index: str,
    query: str, 
    filters: Optional[Dict[str, Any]] = None,
    search_client: OpenSearchClient = None,
    embed_client = None,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 10,
    strategy: str = "enhanced_rrf"
) -> RetrievalResult:
    """
    Universal search tool that wraps adaptive_search_conf functionality.
    
    This tool provides a unified interface to all search strategies and can
    be used as a LangGraph tool for searching different indices.
    
    Args:
        index: Index name to search (e.g., OpenSearchConfig.get_default_index())
        query: Search query text
        filters: Optional filters for search
        search_client: OpenSearch client instance
        embed_client: Embedding client for vector search
        embed_model: Embedding model to use
        top_k: Number of results to return
        strategy: Search strategy ("enhanced_rrf", "bm25", "knn")
        
    Returns:
        RetrievalResult with search results
    """
    try:
        if not search_client:
            raise ValueError("search_client is required")
        
        logger.info(f"Searching index '{index}' with strategy '{strategy}' for query: '{query[:50]}...'")
        
        if strategy == "enhanced_rrf" and embed_client:
            # Enhanced RRF with vector search
            try:
                # Use enhanced embedding utility
                query_embedding = await _create_enhanced_embedding(query, embed_client, embed_model)
                
                # Get expanded candidates from RRF (up to 36)
                rrf_result, diagnostics = await enhanced_rrf_search(
                    query=query,
                    query_embedding=query_embedding,
                    search_client=search_client,
                    index_name=index,
                    filters=filters,
                    top_k=36,  # Expand candidates for cross-encoder
                    use_mmr=True,
                    lambda_param=0.75
                )
                
                # PERFORMANCE FIX: Check coverage on RRF results BEFORE expensive cross-encoder
                # This prevents 7-8s reranking when results will fail coverage anyway
                from src.quality.utils import run_coverage_evaluation
                coverage_result = run_coverage_evaluation(query, rrf_result.results)
                
                if coverage_result["gate_pass"]:
                    # Coverage passed - safe to run expensive cross-encoder reranking
                    from src.services.retrieve import _cross_encoder_rerank
                    reranked_results = _cross_encoder_rerank(
                        query=query,
                        results=rrf_result.results,
                        top_k=4,  # Reduce from 8→4 for cheaper reranking
                        max_rerank_ms=2000   # Add timeout guardrail
                    )
                    method_name = "enhanced_rrf_ce"
                    final_results = reranked_results
                else:
                    # Coverage failed - skip expensive reranking, return RRF results
                    logger.info(f"Coverage gate failed on RRF results (AR={coverage_result['aspect_recall']:.3f}), skipping cross-encoder to save ~7-8s")
                    method_name = "enhanced_rrf_no_ce"
                    final_results = rrf_result.results[:4]  # Consistent with rerank top_k
                
                # Return result with coverage diagnostics
                diagnostics.update({
                    "coverage_gate_pass": coverage_result["gate_pass"],
                    "aspect_recall": coverage_result["aspect_recall"],
                    "alpha_ndcg": coverage_result["alpha_ndcg"],
                    "cross_encoder_skipped": not coverage_result["gate_pass"]
                })
                
                return RetrievalResult(
                    results=final_results,
                    total_found=rrf_result.total_found,
                    retrieval_time_ms=rrf_result.retrieval_time_ms,
                    method=method_name,
                    diagnostics=diagnostics
                )
                
            except EmbeddingError as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")
                # Fall through to BM25
        
        if strategy == "knn" and embed_client:
            # Vector search only
            try:
                # Use enhanced embedding utility
                query_embedding = await _create_enhanced_embedding(query, embed_client, embed_model)
                
                return await knn_search(
                    query_embedding=query_embedding,
                    search_client=search_client,
                    index_name=index,
                    filters=filters,
                    top_k=top_k
                )
                
            except EmbeddingError as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")
                # Fall through to BM25
        
        # BM25 search (default fallback)
        return await bm25_search(
            query=query,
            search_client=search_client,
            index_name=index,
            filters=filters,
            top_k=top_k
        )
        
    except Exception as e:
        logger.error(f"Search tool failed: {e}")
        # Return empty result on failure
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="error",
            diagnostics={"error": str(e)}
        )


async def multi_index_search_tool(
    indices: List[str],
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    search_client: OpenSearchClient = None,
    embed_client = None,
    embed_model: str = "text-embedding-ada-002",
    top_k_per_index: int = 5
) -> List[RetrievalResult]:
    """
    Search multiple indices in parallel and return combined results.
    
    This tool is useful for queries that might span multiple knowledge bases
    (e.g., both confluence and swagger documentation).
    
    Args:
        indices: List of index names to search
        query: Search query text
        filters: Optional filters for search
        search_client: OpenSearch client instance
        embed_client: Embedding client for vector search
        embed_model: Embedding model to use
        top_k_per_index: Number of results per index
        
    Returns:
        List of RetrievalResult objects, one per index
    """
    results = []
    
    for index in indices:
        try:
            result = await search_index_tool(
                index=index,
                query=query,
                filters=filters,
                search_client=search_client,
                embed_client=embed_client,
                embed_model=embed_model,
                top_k=top_k_per_index,
                strategy="enhanced_rrf"
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to search index {index}: {e}")
            # Add empty result to maintain index correspondence
            results.append(RetrievalResult(
                results=[],
                total_found=0,
                retrieval_time_ms=0,
                method="error",
                diagnostics={"error": str(e), "index": index}
            ))
    
    return results


async def adaptive_search_tool(
    query: str,
    intent_confidence: float,
    intent_type: str,
    search_client: OpenSearchClient = None,
    embed_client = None,
    embed_model: str = "text-embedding-ada-002",
    search_index: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    top_k: int = 10
) -> RetrievalResult:
    """
    Adaptive search tool that chooses strategy based on intent and confidence.
    
    This replicates the logic from the original _perform_retrieval function
    but as a reusable LangGraph tool.
    
    Args:
        query: Search query text
        intent_confidence: Confidence score from intent classification
        intent_type: Intent type (confluence, swagger, etc.)
        search_client: OpenSearch client instance
        embed_client: Embedding client for vector search
        embed_model: Embedding model to use
        search_index: Index name to search (use OpenSearchConfig.get_default_index() for main index)
        top_k: Number of results to return
        
    Returns:
        RetrievalResult with search results
    """
    try:
        # Determine index based on intent using centralized config
        if intent_type == "swagger":
            index_name = OpenSearchConfig.get_swagger_index()
        else:
            index_name = search_index or OpenSearchConfig.get_default_index()  # Use provided or default
        
        # Build filters based on intent using centralized config
        filters = OpenSearchConfig.get_intent_filters(intent_type)
        
        # Choose strategy based on intent and confidence
        if intent_type == "list":
            # For list queries, prefer BM25 for broader coverage
            return await search_index_tool(
                index=index_name,
                query=query,
                filters=filters,
                search_client=search_client,
                embed_client=embed_client,
                embed_model=embed_model,
                top_k=top_k,
                strategy="bm25"
            )
        elif intent_confidence > 0.7 and embed_client:
            # High confidence queries use enhanced RRF
            return await search_index_tool(
                index=index_name,
                query=query,
                filters=filters,
                search_client=search_client,
                embed_client=embed_client,
                embed_model=embed_model,
                top_k=top_k,
                strategy="enhanced_rrf"
            )
        else:
            # Low confidence or no embedding client - use BM25
            return await search_index_tool(
                index=index_name,
                query=query,
                filters=filters,
                search_client=search_client,
                embed_client=embed_client,
                embed_model=embed_model,
                top_k=top_k,
                strategy="bm25"
            )
            
    except Exception as e:
        logger.error(f"Adaptive search tool failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="error",
            diagnostics={"error": str(e)}
        )