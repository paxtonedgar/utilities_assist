"""Search tools for LangGraph workflow - wrapping existing search functions."""

import logging
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

from services.retrieve import enhanced_rrf_search, bm25_search, knn_search
from embedding_creation import create_single_embedding, EmbeddingError
from infra.opensearch_client import OpenSearchClient
from src.infra.search_config import OpenSearchConfig
from services.models import RetrievalResult
from src.telemetry.logger import stage

logger = logging.getLogger(__name__)


async def adaptive_search_tool(
    query: str,
    intent_confidence: float,
    intent_type: str,
    search_client: OpenSearchClient,
    embed_client,
    embed_model: str,
    search_index: str,
    top_k: int = 10
) -> RetrievalResult:
    """
    Adaptive search tool that chooses search strategy based on intent and confidence.
    
    This is a wrapper around search_index_tool that provides the adaptive logic
    for choosing between BM25, vector search, and hybrid RRF based on query characteristics.
    
    Args:
        query: Search query text
        intent_confidence: Confidence score from intent classification (0.0-1.0)
        intent_type: Classified intent type (e.g., "confluence", "swagger")
        search_client: OpenSearch client instance
        embed_client: Embedding client for vector search
        embed_model: Embedding model to use
        search_index: Index name to search
        top_k: Number of results to return
        
    Returns:
        RetrievalResult with search results using optimal strategy
    """
    
    # Determine search strategy based on query characteristics
    strategy = "enhanced_rrf"  # Default to hybrid search
    
    # For low confidence intents, use hybrid search for better coverage
    if intent_confidence < 0.6:
        strategy = "enhanced_rrf"
        logger.info(f"Low intent confidence ({intent_confidence:.2f}), using hybrid RRF search")
    
    # For specific technical queries, favor vector search
    elif any(keyword in query.lower() for keyword in ["api", "endpoint", "parameter", "field", "schema"]):
        strategy = "knn" if embed_client else "bm25"
        logger.info(f"Technical query detected, using {'vector' if embed_client else 'BM25'} search")
    
    # For general documentation queries, use hybrid
    else:
        strategy = "enhanced_rrf" if embed_client else "bm25"
        logger.info(f"General query, using {'hybrid RRF' if embed_client else 'BM25'} search")
    
    return await search_index_tool(
        index=search_index,
        query=query,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=top_k,
        strategy=strategy
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
                # Enhanced embedding text with API keywords and domain context
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
                query_embedding = await create_single_embedding(
                    embed_client=embed_client,
                    text=embedding_query,
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
                # Enhanced embedding text with API keywords and domain context
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
                query_embedding = await create_single_embedding(
                    embed_client=embed_client,
                    text=embedding_query,
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
        search_index: Index name to search (e.g., "khub-opensearch-index")
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