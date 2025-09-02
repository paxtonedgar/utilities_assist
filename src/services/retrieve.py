"""Retrieval services for BM25, KNN, and RRF fusion with MMR diversification."""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter  # Fix for MMR diversification

from src.services.models import RetrievalResult, Passage
from src.infra.opensearch_client import OpenSearchClient, SearchFilters
from src.infra.search_config import OpenSearchConfig

# RRF utilities imported within function to avoid dependency loading issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _apply_light_scoring_hints(results: List[Passage]) -> None:
    """Apply light scoring hints as pre-sort reweighting.

    +0.4 if matches step bullets (numbered/bulleted lists)
    +0.3 if jira|servicenow|intake|request (form|ticket)|project key
    +0.2 if heading/title contains verbs (Onboard|Enable|Request|Configure)
    """
    import re

    step_pattern = re.compile(r"(?m)^\s*(\d+[\.\)]|\(\d+\)|[-–•*])\s+\S", re.MULTILINE)
    jira_pattern = re.compile(
        r"\b(jira|servicenow|intake|request\s+(?:form|ticket)|project\s+key)\b",
        re.IGNORECASE,
    )
    verb_pattern = re.compile(r"\b(onboard|enable|request|configure)\b", re.IGNORECASE)

    for result in results:
        bonus = 0.0

        # Check step bullets in content
        if step_pattern.search(result.text):
            bonus += 0.4

        # Check JIRA/ServiceNow terms in content
        if jira_pattern.search(result.text):
            bonus += 0.3

        # Check verbs in title/heading
        title_text = f"{result.title} {result.meta.get('heading', '')}"
        if verb_pattern.search(title_text):
            bonus += 0.2

        # Apply bonus to score
        result.score += bonus


def _cross_encoder_rerank(
    query: str,
    results: List[Passage],
    top_k: int = 8,
    max_rerank_ms: Optional[int] = 15000,
) -> List[Passage]:
    """Apply BGE cross-encoder reranking to RRF candidates.

    Takes up to 36 candidates from RRF, applies cross-encoder scoring,
    and returns top_k results sorted by relevance.
    """
    if not results:
        return []

    try:
        from src.services.reranker import get_reranker
        from src.infra.settings import get_settings

        settings = get_settings()
        if not settings.reranker.enabled:
            logger.info("Cross-encoder disabled, using RRF scores")
            return results[:top_k]

        reranker = get_reranker()
        if not reranker:
            logger.warning("Cross-encoder not available, using RRF scores")
            return results[:top_k]

        # Apply light scoring hints before reranking
        _apply_light_scoring_hints(results)

        # Add timeout guardrail for cross-encoder reranking
        import time

        start_time = time.time() * 1000  # Convert to ms

        try:
            # Rerank with cross-encoder
            reranked_docs, rerank_result = reranker.rerank(
                query=query,
                docs=results,
                min_score=settings.reranker.min_score,
                top_k=top_k,
            )

            # Check if reranking exceeded timeout
            elapsed_ms = (time.time() * 1000) - start_time
            if max_rerank_ms and elapsed_ms > max_rerank_ms:
                logger.warning(
                    f"Cross-encoder timeout ({elapsed_ms:.1f}ms > {max_rerank_ms}ms), falling back to original ranking"
                )
                return results[:top_k]

        except Exception as rerank_error:
            elapsed_ms = (time.time() * 1000) - start_time
            logger.warning(
                f"Cross-encoder failed after {elapsed_ms:.1f}ms: {rerank_error}"
            )
            return results[:top_k]

        logger.info(
            f"Cross-encoder: {len(results)} → {len(reranked_docs)} docs "
            f"(dropped: {rerank_result.dropped_count}, avg_score: {rerank_result.avg_score:.3f}, "
            f"took: {rerank_result.took_ms:.1f}ms)"
        )

        return reranked_docs

    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed: {e}")
        return results[:top_k]


def _create_search_result(client_result) -> Passage:
    """Factory function to create Passage from client result.

    Client result is already a Passage object, just return it directly.
    """
    return client_result


def _create_search_results(client_results) -> List[Passage]:
    """Factory function to create list of Passages from client results."""
    return [_create_search_result(result) for result in client_results]


async def _with_search_timeout(
    search_func,
    stage_name: str,
    timeout_seconds: float,
    method_name: str,
    *args,
    **kwargs,
) -> RetrievalResult:
    """Generic timeout wrapper for all search functions.

    Args:
        search_func: The search function to execute
        stage_name: Name for logging (e.g., 'bm25', 'knn', 'hybrid')
        timeout_seconds: Timeout in seconds
        method_name: Method name for result diagnostics
        *args, **kwargs: Arguments passed to search_func

    Returns:
        RetrievalResult with search results or empty on timeout/error
    """
    try:
        if asyncio.iscoroutinefunction(search_func):
            return await asyncio.wait_for(
                search_func(*args, **kwargs), timeout=timeout_seconds
            )
        else:
            return await asyncio.wait_for(
                asyncio.to_thread(search_func, *args, **kwargs), timeout=timeout_seconds
            )
    except (asyncio.TimeoutError, Exception) as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        error_type = "timeout" if is_timeout else "error"

        # Extract common parameters for logging
        query = kwargs.get("query", args[0] if args else "")
        index_name = kwargs.get("index_name") or kwargs.get("index")
        query_embedding = kwargs.get("query_embedding")

        # STRUCTURED LOGGING: Log timeout/error with telemetry details
        from src.telemetry.logger import log_event

        log_params = {
            "stage": stage_name,
            "event": error_type,
            "err": True,
            "timeout": is_timeout,
            "took_ms": timeout_seconds * 1000,
            "error_type": type(e).__name__,
            "error_message": str(e)[:100],
            "index_name": index_name,
        }

        if query:
            log_params["query_length"] = len(str(query))
        if query_embedding:
            log_params["embedding_dims"] = len(query_embedding)

        log_event(**log_params)

        logger.warning(
            f"{stage_name.upper()} search failed/timed out ({timeout_seconds}s): {type(e).__name__}: {str(e)[:100]}"
        )
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=int(timeout_seconds * 1000),
            method=f"{method_name}_timeout",
            diagnostics={"timeout": is_timeout, "error": str(e)[:100]},
        )


async def bm25_search_with_timeout(
    query: str,
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    time_decay_days: int = 75,
    timeout_seconds: float = 2.0,
) -> RetrievalResult:
    """BM25 search with aggressive timeout and fallback.

    Never propagates exceptions - returns empty results on failure.
    """
    return await _with_search_timeout(
        bm25_search,
        "bm25",
        timeout_seconds,
        "bm25",
        query=query,
        search_client=search_client,
        index_name=index_name,
        filters=filters,
        top_k=top_k,
        time_decay_days=time_decay_days,
    )


async def bm25_search(
    query: str,
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    time_decay_days: int = 75,
) -> RetrievalResult:
    """Perform BM25 text search with enterprise filters and time decay.

    Args:
        query: Search query text
        search_client: OpenSearch client with authentication
        index_name: OpenSearch index name or alias
        filters: ACL, space, and time filters
        top_k: Number of results to return
        time_decay_days: Half-life for time decay in days

    Returns:
        RetrievalResult with BM25 search results
    """
    try:
        # Convert filters dict to SearchFilters
        search_filters = _build_search_filters(filters) if filters else None

        # Perform search
        response = search_client.bm25_search(
            query=query,
            filters=search_filters,
            index=index_name or OpenSearchConfig.get_default_index(),
            k=top_k,
            time_decay_half_life_days=time_decay_days,
        )

        # OpenSearch client returns Passage objects directly - no conversion needed
        results = response.results

        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="bm25",
            diagnostics={"query_type": "bm25", "index": index_name},
        )

    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="bm25",
            diagnostics={"error": str(e), "query_type": "bm25"},
        )


async def knn_search_with_timeout(
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ef_search: int = 256,
    timeout_seconds: float = 2.0,
) -> RetrievalResult:
    """KNN search with aggressive timeout and fallback.

    Never propagates exceptions - returns empty results on failure.
    """
    return await _with_search_timeout(
        knn_search,
        "knn",
        timeout_seconds,
        "knn",
        query_embedding=query_embedding,
        search_client=search_client,
        index_name=index_name,
        filters=filters,
        top_k=top_k,
        ef_search=ef_search,
    )


async def knn_search(
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ef_search: int = 256,
) -> RetrievalResult:
    """Perform KNN vector search with enterprise filters and HNSW optimization.

    Args:
        query_embedding: Query vector embedding (dimensions defined by OpenSearchConfig.EMBEDDING_DIMENSIONS)
        search_client: OpenSearch client with authentication
        index_name: OpenSearch index name or alias (defaults to OpenSearchConfig.get_default_index())
        filters: ACL, space, and time filters
        top_k: Number of results to return
        ef_search: HNSW ef_search parameter (accuracy vs speed trade-off)

    Returns:
        RetrievalResult with KNN search results
    """
    try:
        # Convert filters dict to SearchFilters
        search_filters = _build_search_filters(filters) if filters else None

        # Perform search
        response = search_client.knn_search(
            query_vector=query_embedding,
            filters=search_filters,
            index=index_name or OpenSearchConfig.get_default_index(),
            k=top_k,
            ef_search=ef_search,
        )

        # OpenSearch client returns Passage objects directly
        results = _create_search_results(response.results)

        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="knn",
            diagnostics={
                "query_type": "knn",
                "index": index_name,
                "embedding_dims": len(query_embedding),
            },
        )

    except Exception as e:
        logger.error(f"KNN search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="knn",
            diagnostics={"error": str(e), "query_type": "knn"},
        )


def _build_search_filters(filters: Dict[str, Any]) -> SearchFilters:
    """Convert filter dict to SearchFilters object."""
    return SearchFilters(
        acl_hash=filters.get("acl_hash"),
        space_key=filters.get("space_key"),
        content_type=filters.get("content_type"),
        updated_after=_parse_datetime(filters.get("updated_after")),
        updated_before=_parse_datetime(filters.get("updated_before")),
    )


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime from various formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Failed to parse datetime: {value}")
            return None
    return None


def rrf_fuse_results(
    bm25_hits: List[Tuple[str, float]],
    knn_hits: List[Tuple[str, float]],
    k_final: int = 10,
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    """Enhanced RRF fusion using proven reciprocal rank algorithm.

    Implementation based on standard RRF formula from information retrieval literature.
    More reliable than custom implementation while being simpler than full LangChain integration.

    Args:
        bm25_hits: List of (doc_id, score) from BM25 search
        knn_hits: List of (doc_id, score) from KNN search
        k_final: Number of final results to return
        rrf_k: RRF constant (typically 60)

    Returns:
        List of (doc_id, rrf_score) sorted by RRF score
    """
    # Standard RRF algorithm implementation
    rrf_scores = {}

    # Process BM25 rankings
    for rank, (doc_id, _) in enumerate(bm25_hits):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1.0)

    # Process KNN rankings
    for rank, (doc_id, _) in enumerate(knn_hits):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1.0)

    # Sort by RRF score and return top k
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:k_final]


def lexical_similarity(text1: str, text2: str) -> float:
    """Calculate lexical similarity using scikit-learn's TF-IDF and cosine similarity."""
    if not text1 or not text2:
        return 0.0

    try:
        # Use scikit-learn's TfidfVectorizer for proper TF-IDF computation
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"\b\w+\b",
            stop_words="english",
            max_features=1000,  # Limit features for performance
        )

        # Fit and transform both texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Calculate cosine similarity using scikit-learn
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Return similarity between the two texts
        return float(similarity_matrix[0][1])

    except (ValueError, IndexError):
        # Fallback for edge cases (empty texts, single character, etc.)
        return 0.0


def lexical_relevance(query: str, text: str) -> float:
    """Calculate lexical relevance between query and document text."""
    if not query or not text:
        return 0.0

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    query_tokens = set(tokenize(query))
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    # Simple BM25-like scoring
    text_tf = Counter(text_tokens)
    score = 0.0

    for term in query_tokens:
        if term in text_tf:
            # TF component
            tf = text_tf[term]
            tf_score = tf / (tf + 1.2)  # BM25 k1 parameter
            score += tf_score

    # Normalize by query length
    return score / len(query_tokens)




def _rrf_with_diversification(
    bm25_hits: List[Tuple[str, float]],
    knn_hits: List[Tuple[str, float]],
    all_results: Dict[str, Any],
    query: str,
    k_final: int,
    rrf_k: int = 60,
    lambda_param: float = 0.75,
) -> Tuple[List[str], Dict[str, Any]]:
    """Single-pass RRF fusion with integrated diversification.

    Eliminates redundant MMR processing by combining fusion and diversification.
    """
    # Build rank maps for RRF
    bm25_ranks = {doc_id: idx + 1 for idx, (doc_id, _) in enumerate(bm25_hits)}
    knn_ranks = {doc_id: idx + 1 for idx, (doc_id, _) in enumerate(knn_hits)}

    # Calculate RRF scores
    rrf_scores = {}
    all_doc_ids = set(bm25_ranks.keys()) | set(knn_ranks.keys())

    for doc_id in all_doc_ids:
        rrf_score = 0.0
        if doc_id in bm25_ranks:
            rrf_score += 1.0 / (rrf_k + bm25_ranks[doc_id])
        if doc_id in knn_ranks:
            rrf_score += 1.0 / (rrf_k + knn_ranks[doc_id])
        rrf_scores[doc_id] = rrf_score

    # Sort candidates by RRF score
    candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Single-pass diversification with RRF-aware selection
    selected = []
    diagnostics = {"removed_docs": [], "similarity_scores": {}, "relevance_scores": {}}

    for doc_id, rrf_score in candidates:
        if len(selected) >= k_final:
            break

        if doc_id not in all_results:
            continue

        result = all_results[doc_id]
        doc_text = f"{result.meta.get('title', '')} {result.text}"

        # Calculate relevance to query
        relevance = lexical_relevance(query, doc_text)

        # Calculate maximum similarity to selected documents
        max_similarity = 0.0
        if selected:
            similarities = []
            for selected_doc_id in selected:
                if selected_doc_id in all_results:
                    selected_result = all_results[selected_doc_id]
                    selected_text = f"{selected_result.meta.get('title', '')} {selected_result.text}"
                    sim = lexical_similarity(doc_text, selected_text)
                    similarities.append(sim)
            max_similarity = max(similarities) if similarities else 0.0

        # MMR score with RRF influence: balance relevance, diversity, and RRF ranking
        # Weight RRF score more heavily than in traditional MMR
        mmr_score = (
            lambda_param * relevance
            + 0.3 * rrf_score  # Include RRF ranking in selection
            - (1 - lambda_param) * max_similarity
        )

        # Accept if MMR score is positive (considering all factors)
        if mmr_score > 0 or len(selected) < 3:  # Always take top 3
            selected.append(doc_id)
            diagnostics["relevance_scores"][doc_id] = relevance
            diagnostics["similarity_scores"][doc_id] = max_similarity
        else:
            diagnostics["removed_docs"].append(doc_id)

    logger.info(
        f"Single-pass RRF+MMR: {len(candidates)} candidates → {len(selected)} selected, {len(diagnostics['removed_docs'])} removed"
    )

    return selected, diagnostics


def _hybrid_search_wrapper(
    query: str,
    query_embedding: Optional[List[float]],
    search_client: OpenSearchClient,
    index_name: str = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> RetrievalResult:
    """Sync wrapper for ranx hybrid search to work with timeout decorator."""
    search_filters = _build_search_filters(filters) if filters else None

    # Use ranx RRF hybrid search (industry-standard separate BM25+kNN fusion)
    response = search_client.hybrid_search_ranx(
        query=query,
        query_vector=query_embedding,
        filters=search_filters,
        index=index_name or OpenSearchConfig.get_default_index(),
        k=top_k,
        rrf_k=60,  # Research-proven optimal RRF constant
    )

    # Convert to canonical service format
    results = _create_search_results(response.results)

    return RetrievalResult(
        results=results,
        total_found=response.total_hits,
        retrieval_time_ms=response.took_ms,
        method="hybrid_ranx",
        diagnostics={"query_type": "hybrid_ranx", "index": index_name, "fusion": "ranx_rrf"},
    )


async def hybrid_search_with_timeout(
    query: str,
    query_embedding: Optional[List[float]],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    timeout_seconds: float = 30.0,  # Increased for ranx RRF processing time
) -> RetrievalResult:
    """True hybrid search with aggressive timeout and fallback.

    Uses the new hybrid_search method that combines BM25 and KNN in a single query.
    Never propagates exceptions - returns empty results on failure.
    """
    return await _with_search_timeout(
        _hybrid_search_wrapper,
        "hybrid",
        timeout_seconds,
        "hybrid",
        query=query,
        query_embedding=query_embedding,
        search_client=search_client,
        index_name=index_name,
        filters=filters,
        top_k=top_k,
    )


async def enhanced_rrf_search(
    query: str,
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 8,
    rrf_k: int = 60,
    use_mmr: bool = True,
    lambda_param: float = 0.75,
) -> Tuple[RetrievalResult, Dict[str, Any]]:
    """Enhanced search with RRF fusion and MMR diversification.

    Args:
        query: Search query text
        query_embedding: Query vector embedding
        search_client: OpenSearch client
        index_name: Index name or alias
        filters: Search filters
        top_k: Final number of results
        rrf_k: RRF fusion parameter
        use_mmr: Whether to apply MMR diversification
        lambda_param: MMR trade-off parameter

    Returns:
        Tuple of (RetrievalResult, diagnostics)
    """
    diagnostics = _init_search_diagnostics(use_mmr)

    try:
        # Try hybrid search first, fallback to separate searches
        hybrid_result = await _try_hybrid_search(
            query, query_embedding, search_client, index_name, filters, top_k, diagnostics
        )
        if hybrid_result:
            return hybrid_result

        # Fallback to separate BM25 and KNN searches
        bm25_result, knn_result = await _execute_separate_searches(
            query, query_embedding, search_client, index_name, filters, diagnostics
        )

        # Fuse results and apply diversification
        final_doc_ids, all_results = _fuse_and_diversify_results(
            bm25_result, knn_result, use_mmr, query, top_k, rrf_k, lambda_param, diagnostics
        )

        # Apply no-answer policy and build final results
        return _build_final_search_result(
            final_doc_ids, all_results, bm25_result, knn_result, diagnostics
        )

    except Exception as e:
        logger.error(f"Enhanced RRF search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="enhanced_rrf",
            diagnostics={**diagnostics, "error": str(e)},
        ), {**diagnostics, "error": str(e)}


def _init_search_diagnostics(use_mmr: bool) -> Dict[str, Any]:
    """Initialize search diagnostics dictionary."""
    return {
        "bm25_count": 0,
        "knn_count": 0,
        "rrf_count": 0,
        "mmr_applied": use_mmr,
        "generic_penalties": 0,
        "min_should_match": "70%",
        "decay_scale": "120d",
    }


async def _try_hybrid_search(
    query: str,
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str,
    filters: Optional[Dict[str, Any]],
    top_k: int,
    diagnostics: Dict[str, Any],
) -> Optional[Tuple[RetrievalResult, Dict[str, Any]]]:
    """Try hybrid search first for efficiency."""
    use_single_hybrid = True  # Feature flag - can be configured

    if use_single_hybrid and query_embedding:
        logger.info("Using single hybrid search query")
        hybrid_result = await hybrid_search_with_timeout(
            query=query,
            query_embedding=query_embedding,
            search_client=search_client,
            index_name=index_name,
            filters=filters,
            top_k=top_k,
            timeout_seconds=30.0,
        )

        if hybrid_result.results:
            diagnostics["search_method"] = "single_hybrid"
            diagnostics["result_count"] = len(hybrid_result.results)
            diagnostics["bm25_hits"] = len(hybrid_result.results)  # Approximate
            diagnostics["knn_hits"] = len(hybrid_result.results)  # Approximate
            return hybrid_result, diagnostics
        else:
            logger.info("Hybrid search returned no results, falling back to separate searches")

    return None


async def _execute_separate_searches(
    query: str,
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str,
    filters: Optional[Dict[str, Any]],
    diagnostics: Dict[str, Any],
) -> Tuple[RetrievalResult, RetrievalResult]:
    """Execute separate BM25 and KNN searches with smart gating."""
    from src.infra.settings import get_settings
    settings = get_settings()

    # Run KNN search first
    knn_result = await knn_search_with_timeout(
        query_embedding=query_embedding,
        search_client=search_client,
        index_name=index_name,
        filters=filters,
        top_k=settings.search_config.knn_top_k,
        ef_search=80,
        timeout_seconds=1.8,
    )
    diagnostics["knn_count"] = len(knn_result.results)

    # Decide if we can skip BM25
    bm25_result = await _maybe_skip_bm25_search(
        query, search_client, index_name, filters, knn_result, settings, diagnostics
    )

    return bm25_result, knn_result


async def _maybe_skip_bm25_search(
    query: str,
    search_client: OpenSearchClient,
    index_name: str,
    filters: Optional[Dict[str, Any]],
    knn_result: RetrievalResult,
    settings,
    diagnostics: Dict[str, Any],
) -> RetrievalResult:
    """Skip BM25 if KNN provides sufficient coverage."""
    skip_bm25 = False
    top_k = 8  # From original function parameter

    if len(knn_result.results) >= top_k and knn_result.results:
        avg_knn_score = sum(r.score for r in knn_result.results[:top_k]) / min(
            len(knn_result.results), top_k
        )
        if avg_knn_score > 0.8:  # High semantic similarity threshold
            skip_bm25 = True
            diagnostics["bm25_skipped"] = True
            diagnostics["skip_reason"] = (
                f"KNN sufficient: {len(knn_result.results)} results, avg_score={avg_knn_score:.3f}"
            )
            logger.info(
                f"Skipping BM25 search: KNN returned {len(knn_result.results)} results with avg score {avg_knn_score:.3f}"
            )

    if skip_bm25:
        bm25_result = RetrievalResult(
            results=[], total_found=0, retrieval_time_ms=0, method="bm25_skipped", diagnostics={"skipped": True}
        )
        diagnostics["bm25_count"] = 0
    else:
        bm25_result = await bm25_search_with_timeout(
            query=query,
            search_client=search_client,
            index_name=index_name,
            filters=filters,
            top_k=settings.search_config.bm25_top_k,
            time_decay_days=120,
            timeout_seconds=1.8,
        )
        diagnostics["bm25_count"] = len(bm25_result.results)

    return bm25_result


def _fuse_and_diversify_results(
    bm25_result: RetrievalResult,
    knn_result: RetrievalResult,
    use_mmr: bool,
    query: str,
    top_k: int,
    rrf_k: int,
    lambda_param: float,
    diagnostics: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Fuse results using RRF and optionally apply diversification."""
    bm25_hits = [(r.doc_id, r.score) for r in bm25_result.results]
    knn_hits = [(r.doc_id, r.score) for r in knn_result.results]

    diagnostics["bm25_hits"] = len(bm25_hits)
    diagnostics["knn_hits"] = len(knn_hits)

    all_results = {r.doc_id: r for r in bm25_result.results + knn_result.results}

    if use_mmr:
        final_doc_ids, mmr_diagnostics = _rrf_with_diversification(
            bm25_hits=bm25_hits,
            knn_hits=knn_hits,
            all_results=all_results,
            query=query,
            k_final=36,  # Expand to 36 candidates for cross-encoder
            rrf_k=rrf_k,
            lambda_param=lambda_param,
        )
        diagnostics.update(mmr_diagnostics)
        diagnostics["rrf_count"] = len(final_doc_ids)
    else:
        fused_hits = rrf_fuse_results(bm25_hits, knn_hits, k_final=top_k, rrf_k=rrf_k)
        final_doc_ids = [doc_id for doc_id, _ in fused_hits]
        diagnostics["rrf_count"] = len(fused_hits)

    return final_doc_ids, all_results


def _build_final_search_result(
    final_doc_ids: List[str],
    all_results: Dict[str, Any],
    bm25_result: RetrievalResult,
    knn_result: RetrievalResult,
    diagnostics: Dict[str, Any],
) -> Tuple[RetrievalResult, Dict[str, Any]]:
    """Build final search result with no-answer policy and compression."""
    # Apply no-answer policy
    no_answer_result = _check_no_answer_policy(final_doc_ids, all_results, bm25_result, knn_result, diagnostics)
    if no_answer_result:
        return no_answer_result

    # Build and compress results
    final_results = [all_results[doc_id] for doc_id in final_doc_ids if doc_id in all_results]
    final_results = _compress_results_for_llm(final_results, final_doc_ids, diagnostics)
    _count_generic_penalties(final_results, diagnostics)

    final_result = RetrievalResult(
        results=final_results,
        total_found=len(final_results),
        retrieval_time_ms=bm25_result.retrieval_time_ms + knn_result.retrieval_time_ms,
        method="enhanced_rrf",
        diagnostics=diagnostics,
    )

    return final_result, diagnostics


def _check_no_answer_policy(
    final_doc_ids: List[str],
    all_results: Dict[str, Any],
    bm25_result: RetrievalResult,
    knn_result: RetrievalResult,
    diagnostics: Dict[str, Any],
) -> Optional[Tuple[RetrievalResult, Dict[str, Any]]]:
    """Check if no-answer policy should be applied."""
    total_time = bm25_result.retrieval_time_ms + knn_result.retrieval_time_ms

    if not final_doc_ids:
        logger.info("No documents found - applying no-answer policy")
        return RetrievalResult(
            results=[], total_found=0, retrieval_time_ms=total_time,
            method="enhanced_rrf_no_docs",
            diagnostics={**diagnostics, "no_answer_reason": "no_documents_found"},
        ), {**diagnostics, "no_answer_reason": "no_documents_found"}

    # Check score threshold
    final_results = [all_results[doc_id] for doc_id in final_doc_ids if doc_id in all_results]
    if final_results:
        top_score = max(r.score for r in final_results)
        if top_score < 0.1:
            logger.info(f"Top score {top_score:.3f} below threshold - applying no-answer policy")
            return RetrievalResult(
                results=[], total_found=0, retrieval_time_ms=total_time,
                method="enhanced_rrf_low_score",
                diagnostics={**diagnostics, "no_answer_reason": f"low_score_{top_score:.3f}"},
            ), {**diagnostics, "no_answer_reason": f"low_score_{top_score:.3f}"}

    return None


def _compress_results_for_llm(
    final_results: List[Any], final_doc_ids: List[str], diagnostics: Dict[str, Any]
) -> List[Any]:
    """Compress results for LLM efficiency."""
    MAX_DOCS_FOR_LLM = 5
    MAX_CONTENT_LENGTH = 400

    if len(final_results) > MAX_DOCS_FOR_LLM:
        logger.info(f"Compressing {len(final_results)} docs to top {MAX_DOCS_FOR_LLM}")
        final_results = final_results[:MAX_DOCS_FOR_LLM]
        diagnostics["docs_compressed"] = True
        diagnostics["original_doc_count"] = len(final_doc_ids)

    # Compress content length
    for result in final_results:
        if hasattr(result, "text") and len(result.text) > MAX_CONTENT_LENGTH:
            result.text = result.text[:MAX_CONTENT_LENGTH] + "..."
            diagnostics.setdefault("content_compressed", 0)
            diagnostics["content_compressed"] += 1

    return final_results


def _count_generic_penalties(final_results: List[Any], diagnostics: Dict[str, Any]) -> None:
    """Count generic penalties for diagnostic purposes."""
    for result in final_results:
        section = result.meta.get("section", "").lower()
        title = result.meta.get("title", "").lower()
        if any(generic in section for generic in ["global", "overview", "platform"]) or any(
            generic in title for generic in ["overview", "introduction", "welcome"]
        ):
            diagnostics["generic_penalties"] += 1
