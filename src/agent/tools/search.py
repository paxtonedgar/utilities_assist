"""Search tools for LangGraph workflow - wrapping existing search functions."""

import logging
from typing import Dict, List, Any, Optional

from src.services.retrieve import enhanced_rrf_search, bm25_search, knn_search
from src.embedding_creation import create_single_embedding, EmbeddingError
from src.infra.opensearch_client import OpenSearchClient
from src.infra.search_config import OpenSearchConfig
from src.services.models import RetrievalResult
from src.telemetry.logger import stage
from src.util.cache import ttl_lru

logger = logging.getLogger(__name__)


# Embedding cache for rewritten queries with identical semantics
@ttl_lru(maxsize=1000, ttl_s=600)  # 10 minute cache
def _get_semantic_cache_key(query: str) -> str:
    """
    Create semantic cache key for embedding reuse.

    Normalizes queries to detect semantic equivalence:
    - "What is CIU?" and "what is ciu" → same key
    - "CIU onboarding guide" and "CIU onboarding documentation" → same key
    """
    import re

    # Normalize whitespace and case
    normalized = " ".join(query.lower().split())

    # Remove common variations that don't change semantics
    variations = [
        (r"\bguide\b", "documentation"),
        (r"\bdocs?\b", "documentation"),
        (r"\btutorial\b", "documentation"),
        (r"\bhelp\b", "documentation"),
        (r"\bsetup\b", "configure"),
        (r"\bconfig\b", "configure"),
        (r"\bonboard\b", "onboarding"),
        (r"\bon-board\b", "onboarding"),
    ]

    for pattern, replacement in variations:
        normalized = re.sub(pattern, replacement, normalized)

    return normalized


# Cache embeddings to avoid recomputation for semantically similar queries
_embedding_cache = {}


async def _create_enhanced_embedding(
    query: str, embed_client, embed_model: str
) -> List[float]:
    """Create enhanced embedding with acronym expansion and domain context."""
    # Check embedding cache first
    cache_key = _get_semantic_cache_key(query)
    if cache_key in _embedding_cache:
        logger.debug(f"Embedding cache hit for semantic key: '{cache_key}'")
        return _embedding_cache[cache_key]

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
        embedding = await create_single_embedding(
            embed_client=embed_client,
            text=embedding_query,
            model=embed_model,
            expected_dims=expected_dims,
        )

        # Cache the embedding for reuse
        _embedding_cache[cache_key] = embedding
        logger.debug(f"Cached embedding for semantic key: '{cache_key}'")

        return embedding
    except Exception as e:
        logger.warning(f"Enhanced embedding failed, using simple query: {e}")
        # Fallback to simple query embedding (also cached)
        expected_dims = OpenSearchConfig.EMBEDDING_DIMENSIONS
        embedding = await create_single_embedding(
            embed_client=embed_client,
            text=query,
            model=embed_model,
            expected_dims=expected_dims,
        )

        # Cache fallback embedding too
        _embedding_cache[cache_key] = embedding

        return embedding


@stage("search_execution")
async def search_index_tool(
    index: str,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    search_client: OpenSearchClient = None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 10,
    strategy: str = "enhanced_rrf",
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

        logger.info(
            f"Searching index '{index}' with strategy '{strategy}' for query: '{query[:50]}...'"
        )

        if strategy == "enhanced_rrf" and embed_client:
            # Enhanced RRF with vector search
            try:
                # Use enhanced embedding utility
                query_embedding = await _create_enhanced_embedding(
                    query, embed_client, embed_model
                )

                # Get expanded candidates from RRF (up to 36)
                rrf_result, diagnostics = await enhanced_rrf_search(
                    query=query,
                    query_embedding=query_embedding,
                    search_client=search_client,
                    index_name=index,
                    filters=filters,
                    top_k=36,  # Expand candidates for cross-encoder
                    use_mmr=True,
                    lambda_param=0.75,
                )

                # CRITICAL FIX: Check unique docs in RRF BEFORE expensive cross-encoder
                # Count unique doc_ids to avoid rerank→collapse→coverage fail loop
                unique_docs = len(set(r.doc_id for r in rrf_result.results))
                logger.info(f"RRF unique docs: {unique_docs} (need ≥3 for coverage)")

                if unique_docs >= 3:
                    # Sufficient unique docs - safe to run cross-encoder reranking
                    try:
                        # Import the cross-encoder function directly (avoid import cache issues)
                        from src.services.retrieve import _cross_encoder_rerank

                        logger.info(
                            "Running cross-encoder reranking with timeout guardrail"
                        )
                        reranked_results = _cross_encoder_rerank(
                            query=query,
                            results=rrf_result.results,
                            top_k=4,  # Reduce from 8→4 for cheaper reranking
                            max_rerank_ms=15000,  # CPU BGE reranker needs more time
                        )
                    except TypeError as e:
                        if "max_rerank_ms" in str(e):
                            logger.warning(
                                "Function doesn't support max_rerank_ms yet, using fallback"
                            )
                            from src.services.retrieve import _cross_encoder_rerank

                            reranked_results = _cross_encoder_rerank(
                                query=query, results=rrf_result.results, top_k=4
                            )
                        else:
                            raise

                    # If rerank collapses too much, fall back to RRF results
                    if len(reranked_results) < 2:
                        logger.warning(
                            f"Cross-encoder collapsed to {len(reranked_results)} docs, falling back to RRF top-6"
                        )
                        final_results = rrf_result.results[:6]
                        method_name = "enhanced_rrf_fallback"
                    else:
                        final_results = reranked_results
                        method_name = "enhanced_rrf_ce"
                else:
                    # Insufficient unique docs - skip expensive reranking
                    logger.info(
                        f"Insufficient unique docs ({unique_docs} < 3), skipping cross-encoder to save ~7-8s"
                    )
                    method_name = "enhanced_rrf_no_ce"
                    final_results = rrf_result.results[
                        :6
                    ]  # Return more docs when not reranking

                # Return result with diagnostics
                diagnostics.update(
                    {
                        "unique_docs_pre_rerank": unique_docs,
                        "coverage_check": "unique_doc_count",
                        "cross_encoder_skipped": unique_docs < 3,
                        "method": method_name,
                    }
                )

                return RetrievalResult(
                    results=final_results,
                    total_found=rrf_result.total_found,
                    retrieval_time_ms=rrf_result.retrieval_time_ms,
                    method=method_name,
                    diagnostics=diagnostics,
                )

            except EmbeddingError as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")
                # Fall through to BM25

        if strategy == "knn" and embed_client:
            # Vector search only
            try:
                # Use enhanced embedding utility
                query_embedding = await _create_enhanced_embedding(
                    query, embed_client, embed_model
                )

                return await knn_search(
                    query_embedding=query_embedding,
                    search_client=search_client,
                    index_name=index,
                    filters=filters,
                    top_k=top_k,
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
            top_k=top_k,
        )

    except Exception as e:
        logger.error(f"Search tool failed: {e}")
        # Return empty result on failure
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="error",
            diagnostics={"error": str(e)},
        )


async def multi_index_search_tool(
    indices: List[str],
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    search_client: OpenSearchClient = None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k_per_index: int = 5,
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
                strategy="enhanced_rrf",
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to search index {index}: {e}")
            # Add empty result to maintain index correspondence
            results.append(
                RetrievalResult(
                    results=[],
                    total_found=0,
                    retrieval_time_ms=0,
                    method="error",
                    diagnostics={"error": str(e), "index": index},
                )
            )

    return results


async def adaptive_search_tool(
    query: str,
    intent_confidence: float,
    intent_type: str,
    search_client: OpenSearchClient = None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    search_index: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    top_k: int = 10,
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
            index_name = (
                search_index or OpenSearchConfig.get_default_index()
            )  # Use provided or default

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
                strategy="bm25",
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
                strategy="enhanced_rrf",
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
                strategy="bm25",
            )

    except Exception as e:
        logger.error(f"Adaptive search tool failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="error",
            diagnostics={"error": str(e)},
        )
