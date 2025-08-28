"""Search tools for LangGraph workflow - wrapping existing search functions."""

import logging
from typing import Dict, List, Any, Optional

from src.services.retrieve import enhanced_rrf_search, bm25_search, knn_search, hybrid_search_with_timeout
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
    use_orchestrator: bool = False,
    chat_client=None,
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

        # Apply orchestrator preprocessing if enabled
        filters = await _handle_orchestrator_preprocessing(
            query, filters, use_orchestrator, chat_client
        )
        
        logger.info(
            f"Searching index '{index}' with strategy '{strategy}' for query: '{query[:50]}...'"
        )

        # Execute appropriate search strategy
        if strategy == "enhanced_rrf" and embed_client:
            return await _execute_enhanced_rrf_strategy(
                query, index, filters, search_client, embed_client, embed_model
            )
        elif strategy == "knn" and embed_client:
            return await _execute_knn_strategy(
                query, index, filters, search_client, embed_client, embed_model, top_k
            )
        else:
            # BM25 search (default fallback)
            return await _execute_bm25_fallback(
                query, index, filters, search_client, top_k
            )

    except Exception as e:
        logger.error(f"Search tool failed: {e}")
        return _build_error_result(str(e))


async def unified_content_search_tool(
    content_types: List[str],
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    search_client: OpenSearchClient = None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k_per_type: int = 5,
    unified_index: Optional[str] = None,
) -> List[RetrievalResult]:
    """
    Search unified index with content_type filtering instead of multiple indices.

    Replaces multi-index search with more efficient content_type filtering.
    Uses a single index with content_type metadata for segmentation.

    Args:
        content_types: List of content types to search (e.g., ["confluence", "swagger"])
        query: Search query text
        filters: Optional filters for search
        search_client: OpenSearch client instance
        embed_client: Embedding client for vector search
        embed_model: Embedding model to use
        top_k_per_type: Number of results per content type
        unified_index: Single unified index name (defaults to main index)

    Returns:
        List of RetrievalResult objects, one per content type
    """
    results = []
    
    # Use main index for all content types with filtering
    from src.infra.settings import get_settings
    settings = get_settings()
    index = unified_index or settings.search_index_alias

    for content_type in content_types:
        try:
            # Add content_type filter
            content_filters = filters.copy() if filters else {}
            content_filters["content_type"] = content_type
            
            result = await search_index_tool(
                index=index,
                query=query,
                filters=content_filters,
                search_client=search_client,
                embed_client=embed_client,
                embed_model=embed_model,
                top_k=top_k_per_type,
                strategy="enhanced_rrf",
            )
            
            # Add content type metadata to result
            result.method = f"{result.method}_content_{content_type}"
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to search content_type {content_type}: {e}")
            # Add empty result to maintain content_type correspondence
            results.append(
                RetrievalResult(
                    results=[],
                    total_found=0,
                    retrieval_time_ms=0,
                    method="error",
                    diagnostics={"error": str(e), "content_type": content_type},
                )
            )

    return results


# Backward compatibility alias 
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
    DEPRECATED: Use unified_content_search_tool instead.
    
    Maps index names to content types for backward compatibility:
    - khub-opensearch-index -> confluence
    - khub-opensearch-swagger-index -> swagger
    """
    logger.warning("multi_index_search_tool is deprecated, use unified_content_search_tool")
    
    # Map indices to content types
    content_type_mapping = {
        "khub-opensearch-index": "confluence",
        "khub-opensearch-swagger-index": "swagger",
    }
    
    content_types = []
    for index in indices:
        content_type = content_type_mapping.get(index, "unknown")
        content_types.append(content_type)
    
    return await unified_content_search_tool(
        content_types=content_types,
        query=query,
        filters=filters,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k_per_type=top_k_per_index,
    )


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

        # Get settings for threshold check
        from src.infra.settings import get_settings

        settings = get_settings()

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
        elif (
            intent_confidence > settings.quality_thresholds.high_intent_confidence
            and embed_client
        ):
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


async def _handle_orchestrator_preprocessing(
    query: str, filters: Optional[Dict[str, Any]], use_orchestrator: bool, chat_client
) -> Optional[Dict[str, Any]]:
    """Handle orchestrator preprocessing if enabled."""
    if not (use_orchestrator and chat_client):
        return filters
    
    try:
        from src.agent.orchestrator import StreamlinedOrchestrator
        
        orchestrator = StreamlinedOrchestrator(chat_client)
        plan = orchestrator._llm_plan(query)
        
        # Apply planned filters
        if plan.steps and plan.steps[0].get("filters"):
            updated_filters = {**(filters or {}), **plan.steps[0]["filters"]}
            logger.info(f"Orchestrator added filters: {plan.steps[0]['filters']}")
            return updated_filters
    except Exception as e:
        logger.debug(f"Orchestrator preprocessing skipped: {e}")
    
    return filters


async def _execute_enhanced_rrf_strategy(
    query: str, index: str, filters: Optional[Dict[str, Any]], 
    search_client: OpenSearchClient, embed_client, embed_model: str
) -> RetrievalResult:
    """Execute native hybrid search strategy (cross-encoder reranking disabled for performance)."""
    try:
        # Create embedding for vector search
        query_embedding = await _create_enhanced_embedding(query, embed_client, embed_model)
        
        # Use native hybrid search directly (no custom RRF needed)
        rrf_result, diagnostics = await _get_native_hybrid_results(
            query, query_embedding, search_client, index, filters
        )
        
        # DISABLED: Cross-encoder reranking (saves 3.6s latency)
        # Native OpenSearch RRF provides good relevance without additional ML overhead
        unique_docs = len(set(r.doc_id for r in rrf_result.results))
        final_results = rrf_result.results
        method_name = "hybrid_native_no_rerank"
        
        logger.info(f"Native hybrid search: {len(final_results)} results, {unique_docs} unique docs (no reranking)")
        
        # Build final result with diagnostics
        return _build_enhanced_rrf_result(
            final_results, rrf_result, method_name, diagnostics, unique_docs
        )
        
    except EmbeddingError as e:
        logger.warning(f"Embedding failed, falling back to BM25: {e}")
        return await _execute_bm25_fallback(query, index, filters, search_client, 10)


async def _execute_knn_strategy(
    query: str, index: str, filters: Optional[Dict[str, Any]],
    search_client: OpenSearchClient, embed_client, embed_model: str, top_k: int
) -> RetrievalResult:
    """Execute KNN vector search strategy."""
    try:
        query_embedding = await _create_enhanced_embedding(query, embed_client, embed_model)
        
        return await knn_search(
            query_embedding=query_embedding,
            search_client=search_client,
            index_name=index,
            filters=filters,
            top_k=top_k,
        )
    except EmbeddingError as e:
        logger.warning(f"Embedding failed, falling back to BM25: {e}")
        return await _execute_bm25_fallback(query, index, filters, search_client, top_k)


async def _execute_bm25_fallback(
    query: str, index: str, filters: Optional[Dict[str, Any]], 
    search_client: OpenSearchClient, top_k: int
) -> RetrievalResult:
    """Execute BM25 search as fallback strategy."""
    return await bm25_search(
        query=query,
        search_client=search_client,
        index_name=index,
        filters=filters,
        top_k=top_k,
    )


async def _get_rrf_results(
    query: str, query_embedding, search_client: OpenSearchClient, 
    index: str, filters: Optional[Dict[str, Any]]
):
    """Get RRF search results with configurable parameters."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    return await enhanced_rrf_search(
        query=query,
        query_embedding=query_embedding,
        search_client=search_client,
        index_name=index,
        filters=filters,
        top_k=settings.search_config.rrf_expansion_candidates,
        use_mmr=False,  # MMR removed to save latency
        lambda_param=settings.search_config.rrf_lambda_param,
    )


async def _get_native_hybrid_results(
    query: str, query_embedding, search_client: OpenSearchClient, 
    index: str, filters: Optional[Dict[str, Any]]
):
    """Get native OpenSearch hybrid search results."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    # Use native hybrid search wrapper (already updated to use hybrid_search_native)
    hybrid_result = await hybrid_search_with_timeout(
        query=query,
        query_embedding=query_embedding,
        search_client=search_client,
        index_name=index,
        filters=filters,
        top_k=settings.search_config.rrf_expansion_candidates,
        timeout_seconds=5.0,
    )
    
    # Return in same format as enhanced_rrf_search for compatibility
    return hybrid_result, {"search_method": "native_hybrid", "result_count": len(hybrid_result.results)}


def _should_apply_cross_encoder_reranking(rrf_result) -> tuple[bool, int]:
    """Determine if cross-encoder reranking should be applied."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    unique_docs = len(set(r.doc_id for r in rrf_result.results))
    min_docs_threshold = settings.quality_thresholds.min_unique_docs
    
    logger.info(f"RRF unique docs: {unique_docs} (need ≥{min_docs_threshold} for coverage)")
    
    return unique_docs >= min_docs_threshold, unique_docs


async def _apply_cross_encoder_reranking(query: str, rrf_result) -> tuple[list, str]:
    """Apply cross-encoder reranking with fallback handling."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    try:
        from src.services.retrieve import _cross_encoder_rerank
        
        logger.info("Running cross-encoder reranking with timeout guardrail")
        reranked_results = _cross_encoder_rerank(
            query=query,
            results=rrf_result.results,
            top_k=settings.search_config.rerank_top_k,
            max_rerank_ms=settings.search_config.rerank_timeout_ms,
        )
    except TypeError as e:
        if "max_rerank_ms" in str(e):
            logger.warning("Function doesn't support max_rerank_ms yet, using fallback")
            reranked_results = _cross_encoder_rerank(
                query=query,
                results=rrf_result.results,
                top_k=settings.reranker.top_k,
            )
        else:
            raise
    
    # Check for reranking collapse
    min_rerank_threshold = 3
    if len(reranked_results) < min_rerank_threshold:
        logger.warning(
            f"Cross-encoder collapsed to {len(reranked_results)} docs, "
            f"falling back to RRF top-{settings.search_config.rrf_unique_limit}"
        )
        final_results = rrf_result.results[:settings.search_config.rrf_unique_limit]
        method_name = "enhanced_rrf_fallback"
    else:
        final_results = reranked_results
        method_name = "enhanced_rrf_ce"
    
    return final_results, method_name


def _skip_cross_encoder_reranking(rrf_result, unique_docs: int) -> tuple[list, str]:
    """Skip cross-encoder reranking due to insufficient unique docs."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    min_docs_threshold = settings.quality_thresholds.min_unique_docs
    logger.info(
        f"Insufficient unique docs ({unique_docs} < {min_docs_threshold}), "
        "skipping cross-encoder to save ~7-8s"
    )
    
    final_results = rrf_result.results[:settings.search_config.rrf_unique_limit]
    method_name = "enhanced_rrf_no_ce"
    
    return final_results, method_name


def _build_enhanced_rrf_result(
    final_results: list, rrf_result, method_name: str, 
    diagnostics: dict, unique_docs: int
) -> RetrievalResult:
    """Build final enhanced RRF result with diagnostics."""
    from src.infra.settings import get_settings
    settings = get_settings()
    
    diagnostics.update({
        "unique_docs_pre_rerank": unique_docs,
        "coverage_check": "unique_doc_count",
        "cross_encoder_skipped": unique_docs < settings.quality_thresholds.min_unique_docs,
        "method": method_name,
    })
    
    return RetrievalResult(
        results=final_results,
        total_found=rrf_result.total_found,
        retrieval_time_ms=rrf_result.retrieval_time_ms,
        method=method_name,
        diagnostics=diagnostics,
    )


def _build_error_result(error_message: str) -> RetrievalResult:
    """Build error result for failed searches."""
    return RetrievalResult(
        results=[],
        total_found=0,
        retrieval_time_ms=0,
        method="error",
        diagnostics={"error": error_message},
    )


# Bridge functions for the simplified search architecture
class SearchOptions:
    """Options for customizing search behavior."""
    def __init__(self, content_boost=None, content_filter=None, top_k=15):
        self.content_boost = content_boost
        self.content_filter = content_filter
        self.top_k = top_k


class ContentType:
    """Content type classifications."""
    API_SPEC = "api_spec"
    RUNBOOK = "runbook"
    LIST_DATA = "list_data"


class UnifiedSearchResult:
    """Unified search result container."""
    def __init__(self, passages, total_found, search_strategy):
        self.passages = passages
        self.total_found = total_found
        self.search_strategy = search_strategy


async def search_docs(query: str, search_client, embed_client, embed_model: str, options=None, filters: Optional[Dict[str, Any]] = None):
    """Bridge function for unified search."""
    from src.infra.search_config import OpenSearchConfig
    
    result = await search_index_tool(
        index=OpenSearchConfig.get_default_index(),
        query=query,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=options.top_k if options else 15,
        filters=filters,
    )
    
    return UnifiedSearchResult(
        passages=result.results,
        total_found=result.total_found,
        search_strategy=result.method
    )


async def search_api_docs(query: str, search_client, embed_client, embed_model: str, filters: Optional[Dict[str, Any]] = None):
    """Search API documentation."""
    from src.infra.search_config import OpenSearchConfig
    
    result = await search_index_tool(
        index=OpenSearchConfig.get_swagger_index(),
        query=query,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=15,
        filters=filters,
    )
    
    return UnifiedSearchResult(
        passages=result.results,
        total_found=result.total_found,
        search_strategy="api_focused"
    )


async def search_procedures(query: str, search_client, embed_client, embed_model: str, filters: Optional[Dict[str, Any]] = None):
    """Search procedures and runbooks."""
    from src.infra.search_config import OpenSearchConfig
    
    result = await search_index_tool(
        index=OpenSearchConfig.get_default_index(),
        query=query,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=15,
        strategy="enhanced_rrf",
        filters=filters,
    )
    
    return UnifiedSearchResult(
        passages=result.results,
        total_found=result.total_found,
        search_strategy="procedure_focused"
    )


async def search_general(query: str, search_client, embed_client, embed_model: str, filters: Optional[Dict[str, Any]] = None):
    """General search without specific focus."""
    from src.infra.search_config import OpenSearchConfig
    
    result = await search_index_tool(
        index=OpenSearchConfig.get_default_index(),
        query=query,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=15,
        filters=filters,
    )
    
    return UnifiedSearchResult(
        passages=result.results,
        total_found=result.total_found,
        search_strategy="general"
    )
