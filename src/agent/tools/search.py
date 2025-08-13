"""Search tools for LangGraph workflow - wrapping existing search functions."""

import logging
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

from services.retrieve import enhanced_rrf_search, bm25_search, knn_search
from embedding_creation import create_single_embedding, EmbeddingError
from infra.opensearch_client import OpenSearchClient
from services.models import RetrievalResult

logger = logging.getLogger(__name__)


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
        index: Index name to search (e.g., "confluence_current", "swagger", etc.)
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
                expected_dims = 1536
                query_embedding = await create_single_embedding(
                    embed_client=embed_client,
                    text=query,
                    model=embed_model,
                    expected_dims=expected_dims
                )
                
                result, diagnostics = await enhanced_rrf_search(
                    query=query,
                    query_embedding=query_embedding,
                    search_client=search_client,
                    index_name=index,
                    filters=filters,
                    top_k=top_k,
                    use_mmr=True,
                    lambda_param=0.75
                )
                return result
                
            except EmbeddingError as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")
                # Fall through to BM25
        
        if strategy == "knn" and embed_client:
            # Vector search only
            try:
                expected_dims = 1536
                query_embedding = await create_single_embedding(
                    embed_client=embed_client,
                    text=query,
                    model=embed_model,
                    expected_dims=expected_dims
                )
                
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
    use_mock_corpus: bool = False,
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
        use_mock_corpus: Whether to use mock index
        top_k: Number of results to return
        
    Returns:
        RetrievalResult with search results
    """
    try:
        # Determine index based on intent
        if use_mock_corpus:
            index_name = "confluence_mock"
        elif intent_type == "swagger":
            index_name = "khub-opensearch-swagger-index"
        else:
            index_name = "confluence_current"  # Default index
        
        # Build filters based on intent
        filters = {}
        if intent_type == "confluence":
            filters["content_type"] = "confluence"
        elif intent_type == "swagger":
            filters["content_type"] = "api_spec"
        
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