"""Retrieval services for BM25, KNN, and RRF fusion with MMR diversification."""

import logging
import time
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter  # Fix for MMR diversification

from services.models import SearchResult, RetrievalResult
from infra.opensearch_client import OpenSearchClient, SearchFilters
from src.infra.search_config import OpenSearchConfig
# RRF utilities imported within function to avoid dependency loading issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


async def bm25_search_with_timeout(
    query: str,
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    time_decay_days: int = 75,
    timeout_seconds: float = 2.0
) -> RetrievalResult:
    """BM25 search with aggressive timeout and fallback.
    
    Never propagates exceptions - returns empty results on failure.
    """
    try:
        return await asyncio.wait_for(
            bm25_search(
                query=query,
                search_client=search_client,
                index_name=index_name,
                filters=filters,
                top_k=top_k,
                time_decay_days=time_decay_days
            ),
            timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, Exception) as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        error_type = "timeout" if is_timeout else "error"
        
        # STRUCTURED LOGGING: Log timeout/error with telemetry details
        from src.telemetry.logger import log_event
        log_event(
            stage="bm25",
            event=error_type,
            err=True,
            timeout=is_timeout,
            took_ms=timeout_seconds * 1000,
            error_type=type(e).__name__,
            error_message=str(e)[:100],
            index_name=index_name,
            query_length=len(query)
        )
        
        logger.warning(f"BM25 search failed/timed out ({timeout_seconds}s): {type(e).__name__}: {str(e)[:100]}")
        return RetrievalResult(
            results=[], total_found=0, retrieval_time_ms=int(timeout_seconds * 1000),
            method="bm25_timeout", diagnostics={"timeout": is_timeout, "error": str(e)[:100]}
        )


async def bm25_search(
    query: str,
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    time_decay_days: int = 75
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
            time_decay_half_life_days=time_decay_days
        )
        
        # Convert to canonical service format - OpenSearch client now returns services.models.SearchResult
        results = []
        for result in response.results:
            # Result is already a services.models.SearchResult with canonical schema
            # Create canonical SearchResult ensuring all required fields are present
            service_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,     # Now available in canonical schema
                url=result.url,         # Now available in canonical schema
                score=result.score,
                content=result.content, # Use content field consistently
                metadata=result.metadata
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="bm25",
            diagnostics={"query_type": "bm25", "index": index_name}
        )
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="bm25",
            diagnostics={"error": str(e), "query_type": "bm25"}
        )


async def knn_search_with_timeout(
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ef_search: int = 256,
    timeout_seconds: float = 2.0
) -> RetrievalResult:
    """KNN search with aggressive timeout and fallback.
    
    Never propagates exceptions - returns empty results on failure.
    """
    try:
        return await asyncio.wait_for(
            knn_search(
                query_embedding=query_embedding,
                search_client=search_client,
                index_name=index_name,
                filters=filters,
                top_k=top_k,
                ef_search=ef_search
            ),
            timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, Exception) as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        error_type = "timeout" if is_timeout else "error"
        
        # STRUCTURED LOGGING: Log timeout/error with telemetry details
        from src.telemetry.logger import log_event
        log_event(
            stage="knn",
            event=error_type,
            err=True,
            timeout=is_timeout,
            took_ms=timeout_seconds * 1000,
            error_type=type(e).__name__,
            error_message=str(e)[:100],
            index_name=index_name,
            embedding_dims=len(query_embedding)
        )
        
        logger.warning(f"KNN search failed/timed out ({timeout_seconds}s): {type(e).__name__}: {str(e)[:100]}")
        return RetrievalResult(
            results=[], total_found=0, retrieval_time_ms=int(timeout_seconds * 1000),
            method="knn_timeout", diagnostics={"timeout": is_timeout, "error": str(e)[:100]}
        )


async def knn_search(
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ef_search: int = 256
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
            ef_search=ef_search
        )
        
        # Convert to canonical service format - OpenSearch client now returns services.models.SearchResult  
        results = []
        for result in response.results:
            # Result is already a services.models.SearchResult with canonical schema
            # Create canonical SearchResult ensuring all required fields are present
            service_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,     # Now available in canonical schema
                url=result.url,         # Now available in canonical schema
                score=result.score,
                content=result.content, # Use content field consistently
                metadata=result.metadata
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="knn",
            diagnostics={"query_type": "knn", "index": index_name, "embedding_dims": len(query_embedding)}
        )
        
    except Exception as e:
        logger.error(f"KNN search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="knn",
            diagnostics={"error": str(e), "query_type": "knn"}
        )


async def rrf_fuse(
    bm25_result: RetrievalResult,
    knn_result: RetrievalResult,
    search_client: OpenSearchClient,
    top_k: int = 8,
    rrf_k: int = 60
) -> RetrievalResult:
    """Fuse BM25 and KNN results using Reciprocal Rank Fusion.
    
    Uses enterprise-grade RRF implementation from OpenSearchClient.
    
    Args:
        bm25_result: Results from BM25 search
        knn_result: Results from KNN search
        search_client: OpenSearch client for RRF fusion
        top_k: Number of final results to return
        rrf_k: RRF constant (typically 60)
        
    Returns:
        RetrievalResult with fused and re-ranked results
    """
    try:
        # Convert service results to client format for fusion
        from infra.opensearch_client import SearchResponse
        from services.models import SearchResult as ServiceSearchResult  # Use the service model that RRF expects
        
        bm25_response = SearchResponse(
            results=[
                ServiceSearchResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    content=r.content,  # Use content field consistently
                    metadata={
                        **r.metadata,
                        "title": r.metadata.get("title", "")  # Ensure title is available
                    }
                ) for r in bm25_result.results
            ],
            total_hits=bm25_result.total_found,
            took_ms=bm25_result.retrieval_time_ms,
            method="bm25"
        )
        
        knn_response = SearchResponse(
            results=[
                ServiceSearchResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    content=r.content,  # Use content field consistently
                    metadata={
                        **r.metadata,
                        "title": r.metadata.get("title", "")  # Ensure title is available
                    }
                ) for r in knn_result.results
            ],
            total_hits=knn_result.total_found,
            took_ms=knn_result.retrieval_time_ms,
            method="knn"
        )
        
        # Perform RRF fusion
        fused_response = search_client.rrf_fuse(
            bm25_response=bm25_response,
            knn_response=knn_response,
            k=top_k,
            rrf_k=rrf_k
        )
        
        # Convert back to canonical service format - RRF fusion returns services.models.SearchResult
        results = []
        for result in fused_response.results:
            # Result is already a services.models.SearchResult with canonical schema from RRF fusion
            # Create canonical SearchResult ensuring all required fields are present
            service_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,     # Now available in canonical schema
                url=result.url,         # Now available in canonical schema
                score=result.score,
                content=result.content, # Use content field consistently
                metadata=result.metadata
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=len(results),
            retrieval_time_ms=fused_response.took_ms,
            method="rrf",
            diagnostics={"query_type": "rrf", "bm25_count": len(bm25_result.results), "knn_count": len(knn_result.results)}
        )
        
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="rrf",
            diagnostics={"error": str(e), "query_type": "rrf"}
        )


def _build_search_filters(filters: Dict[str, Any]) -> SearchFilters:
    """Convert filter dict to SearchFilters object."""
    return SearchFilters(
        acl_hash=filters.get("acl_hash"),
        space_key=filters.get("space_key"),
        content_type=filters.get("content_type"),
        updated_after=_parse_datetime(filters.get("updated_after")),
        updated_before=_parse_datetime(filters.get("updated_before"))
    )


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime from various formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Failed to parse datetime: {value}")
            return None
    return None


def rrf_fuse_results(
    bm25_hits: List[Tuple[str, float]], 
    knn_hits: List[Tuple[str, float]], 
    k_final: int = 10, 
    rrf_k: int = 60
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
            token_pattern=r'\b\w+\b',
            stop_words='english',
            max_features=1000  # Limit features for performance
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
        return re.findall(r'\b\w+\b', text.lower())
    
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


def mmr_diversify(
    candidates: List[str], 
    doc_text_lookup: Dict[str, str], 
    query: str, 
    k: int = 8, 
    lambda_param: float = 0.75
) -> Tuple[List[str], Dict[str, Any]]:
    """MMR diversification to reduce redundant results.
    
    Args:
        candidates: List of doc_ids sorted by fused score
        doc_text_lookup: Mapping from doc_id to text content
        query: Original search query
        k: Number of results to select
        lambda_param: Trade-off between relevance and diversity (0.0-1.0)
        
    Returns:
        Tuple of (selected_doc_ids, diagnostics)
    """
    selected = []
    remaining = candidates.copy()
    diagnostics = {
        "removed_docs": [],
        "similarity_scores": {},
        "relevance_scores": {}
    }
    
    while remaining and len(selected) < k:
        best_doc = None
        best_score = -float('inf')
        best_rel = 0.0
        best_div = 0.0
        
        for doc_id in remaining:
            # Get document text, fallback to doc_id if not found
            doc_text = doc_text_lookup.get(doc_id, doc_id)
            
            # Calculate relevance to query
            relevance = lexical_relevance(query, doc_text)
            
            # Calculate maximum similarity to already selected documents
            max_similarity = 0.0
            if selected:
                similarities = []
                for selected_doc in selected:
                    selected_text = doc_text_lookup.get(selected_doc, selected_doc)
                    sim = lexical_similarity(doc_text, selected_text)
                    similarities.append(sim)
                max_similarity = max(similarities)
            
            # MMR score: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_score:
                best_doc = doc_id
                best_score = mmr_score
                best_rel = relevance
                best_div = max_similarity
        
        if best_doc:
            selected.append(best_doc)
            remaining.remove(best_doc)
            
            diagnostics["relevance_scores"][best_doc] = best_rel
            diagnostics["similarity_scores"][best_doc] = best_div
    
    # Track removed documents
    diagnostics["removed_docs"] = [doc for doc in candidates if doc not in selected]
    
    logger.info(f"MMR diversification: {len(candidates)} candidates → {len(selected)} selected, {len(diagnostics['removed_docs'])} removed")
    
    return selected, diagnostics


def _rrf_with_diversification(
    bm25_hits: List[Tuple[str, float]],
    knn_hits: List[Tuple[str, float]], 
    all_results: Dict[str, Any],
    query: str,
    k_final: int,
    rrf_k: int = 60,
    lambda_param: float = 0.75
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
        doc_text = f"{result.metadata.get('title', '')} {result.content}"
        
        # Calculate relevance to query
        relevance = lexical_relevance(query, doc_text)
        
        # Calculate maximum similarity to selected documents  
        max_similarity = 0.0
        if selected:
            similarities = []
            for selected_doc_id in selected:
                if selected_doc_id in all_results:
                    selected_result = all_results[selected_doc_id]
                    selected_text = f"{selected_result.metadata.get('title', '')} {selected_result.content}"
                    sim = lexical_similarity(doc_text, selected_text)
                    similarities.append(sim)
            max_similarity = max(similarities) if similarities else 0.0
        
        # MMR score with RRF influence: balance relevance, diversity, and RRF ranking
        # Weight RRF score more heavily than in traditional MMR
        mmr_score = (lambda_param * relevance + 
                     0.3 * rrf_score -  # Include RRF ranking in selection 
                     (1 - lambda_param) * max_similarity)
        
        # Accept if MMR score is positive (considering all factors)
        if mmr_score > 0 or len(selected) < 3:  # Always take top 3
            selected.append(doc_id)
            diagnostics["relevance_scores"][doc_id] = relevance
            diagnostics["similarity_scores"][doc_id] = max_similarity
        else:
            diagnostics["removed_docs"].append(doc_id)
    
    logger.info(f"Single-pass RRF+MMR: {len(candidates)} candidates → {len(selected)} selected, {len(diagnostics['removed_docs'])} removed")
    
    return selected, diagnostics


async def hybrid_search_with_timeout(
    query: str,
    query_embedding: Optional[List[float]],
    search_client: OpenSearchClient,
    index_name: str = None,  # Will use OpenSearchConfig.get_default_index() if None
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    timeout_seconds: float = 3.0
) -> RetrievalResult:
    """True hybrid search with aggressive timeout and fallback.
    
    Uses the new hybrid_search method that combines BM25 and KNN in a single query.
    Never propagates exceptions - returns empty results on failure.
    """
    try:
        # Convert filters dict to SearchFilters
        search_filters = _build_search_filters(filters) if filters else None
        
        # Execute hybrid search with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                search_client.hybrid_search,
                query=query,
                query_vector=query_embedding,
                filters=search_filters,
                index=index_name or OpenSearchConfig.get_default_index(),
                k=top_k
            ),
            timeout=timeout_seconds
        )
        
        # Convert to canonical service format
        results = []
        for result in response.results:
            service_result = SearchResult(
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                score=result.score,
                content=result.content,
                metadata=result.metadata
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="hybrid",
            diagnostics={"query_type": "hybrid", "index": index_name}
        )
        
    except (asyncio.TimeoutError, Exception) as e:
        is_timeout = isinstance(e, asyncio.TimeoutError)
        error_type = "timeout" if is_timeout else "error"
        
        # STRUCTURED LOGGING: Log timeout/error with telemetry details
        from src.telemetry.logger import log_event
        log_event(
            stage="hybrid",
            event=error_type,
            err=True,
            timeout=is_timeout,
            took_ms=timeout_seconds * 1000,
            error_type=type(e).__name__,
            error_message=str(e)[:100],
            index_name=index_name,
            query_length=len(query)
        )
        
        logger.warning(f"Hybrid search failed/timed out ({timeout_seconds}s): {type(e).__name__}: {str(e)[:100]}")
        return RetrievalResult(
            results=[], total_found=0, retrieval_time_ms=int(timeout_seconds * 1000),
            method="hybrid_timeout", diagnostics={"timeout": is_timeout, "error": str(e)[:100]}
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
    lambda_param: float = 0.75
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
    diagnostics = {
        "bm25_count": 0,
        "knn_count": 0,
        "rrf_count": 0,
        "mmr_applied": use_mmr,
        "generic_penalties": 0,
        "min_should_match": "70%",
        "decay_scale": "120d"
    }
    
    try:
        # OPTION 1: Try true hybrid search first (single query with both BM25 and KNN)
        # This is more efficient than running separate searches
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
                timeout_seconds=2.5  # Slightly longer timeout for hybrid
            )
            
            if hybrid_result.results:
                # Success with hybrid search - return directly
                diagnostics["search_method"] = "single_hybrid"
                diagnostics["result_count"] = len(hybrid_result.results)
                
                return hybrid_result, diagnostics
            else:
                logger.info("Hybrid search returned no results, falling back to separate searches")
        
        # OPTION 2: Fallback to separate BM25 and KNN searches with RRF fusion
        # AGGRESSIVE TIMEOUTS: Use timeout-wrapped search calls that never propagate exceptions
        # This prevents the 6-7s "Answer" with zero docs scenario
        
        knn_result = await knn_search_with_timeout(
            query_embedding=query_embedding,
            search_client=search_client,
            index_name=index_name,
            filters=filters,
            top_k=50,  # Get more candidates for better fusion
            ef_search=256,
            timeout_seconds=1.8  # Aggressive timeout
        )
        
        diagnostics["knn_count"] = len(knn_result.results)
        
        # Gate logic: Skip BM25 if KNN already provides good coverage
        skip_bm25 = False
        if len(knn_result.results) >= top_k and knn_result.results:
            # Check if KNN results have high scores (good semantic match)
            avg_knn_score = sum(r.score for r in knn_result.results[:top_k]) / min(len(knn_result.results), top_k)
            if avg_knn_score > 0.8:  # High semantic similarity threshold
                skip_bm25 = True
                diagnostics["bm25_skipped"] = True
                diagnostics["skip_reason"] = f"KNN sufficient: {len(knn_result.results)} results, avg_score={avg_knn_score:.3f}"
                logger.info(f"Skipping BM25 search: KNN returned {len(knn_result.results)} results with avg score {avg_knn_score:.3f}")
        
        if skip_bm25:
            # Use only KNN results - no BM25 search needed
            bm25_result = RetrievalResult(
                results=[], total_found=0, retrieval_time_ms=0, 
                method="bm25_skipped", diagnostics={"skipped": True}
            )
            diagnostics["bm25_count"] = 0
        else:
            # Run BM25 search with timeout protection  
            bm25_result = await bm25_search_with_timeout(
                query=query,
                search_client=search_client,
                index_name=index_name,
                filters=filters,
                top_k=50,  # Get more candidates for better fusion
                time_decay_days=120,
                timeout_seconds=1.8  # Aggressive timeout
            )
            diagnostics["bm25_count"] = len(bm25_result.results)
        
        # Convert to (doc_id, score) tuples for RRF
        bm25_hits = [(r.doc_id, r.score) for r in bm25_result.results]
        knn_hits = [(r.doc_id, r.score) for r in knn_result.results]
        
        # SINGLE-PASS DIVERSIFICATION: Combine RRF fusion with diversification in one step
        # This eliminates redundant MMR processing after multiple fusion steps
        all_results = {r.doc_id: r for r in bm25_result.results + knn_result.results}
        
        if use_mmr:
            # Apply RRF with integrated diversification
            final_doc_ids, mmr_diagnostics = _rrf_with_diversification(
                bm25_hits=bm25_hits,
                knn_hits=knn_hits,
                all_results=all_results,
                query=query,
                k_final=top_k,
                rrf_k=rrf_k,
                lambda_param=lambda_param
            )
            diagnostics.update(mmr_diagnostics)
            diagnostics["rrf_count"] = len(final_doc_ids)
        else:
            # Traditional RRF without diversification
            fused_hits = rrf_fuse_results(bm25_hits, knn_hits, k_final=top_k, rrf_k=rrf_k)
            final_doc_ids = [doc_id for doc_id, _ in fused_hits]
            diagnostics["rrf_count"] = len(fused_hits)
        
        # NO-ANSWER POLICY: Early exit if no docs or low scores
        # This prevents 6-7s LLM calls with empty context
        if not final_doc_ids:
            logger.info("No documents found - applying no-answer policy")
            return RetrievalResult(
                results=[],
                total_found=0,
                retrieval_time_ms=bm25_result.retrieval_time_ms + knn_result.retrieval_time_ms,
                method="enhanced_rrf_no_docs",
                diagnostics={**diagnostics, "no_answer_reason": "no_documents_found"}
            ), {**diagnostics, "no_answer_reason": "no_documents_found"}
        
        # Build final results
        final_results = []
        for doc_id in final_doc_ids:
            if doc_id in all_results:
                final_results.append(all_results[doc_id])
        
        # Check score threshold after building results
        if final_results:
            top_score = max(r.score for r in final_results)
            if top_score < 0.1:  # Low confidence threshold
                logger.info(f"Top score {top_score:.3f} below threshold - applying no-answer policy")
                return RetrievalResult(
                    results=[],
                    total_found=0,
                    retrieval_time_ms=bm25_result.retrieval_time_ms + knn_result.retrieval_time_ms,
                    method="enhanced_rrf_low_score",
                    diagnostics={**diagnostics, "no_answer_reason": f"low_score_{top_score:.3f}"}
                ), {**diagnostics, "no_answer_reason": f"low_score_{top_score:.3f}"}
        
        # MAX DOCS COMPRESSION: Limit to best 3-5 chunks (~1-2k tokens)
        MAX_DOCS_FOR_LLM = 5
        MAX_CONTENT_LENGTH = 400  # ~100 tokens per chunk
        
        if len(final_results) > MAX_DOCS_FOR_LLM:
            logger.info(f"Compressing {len(final_results)} docs to top {MAX_DOCS_FOR_LLM}")
            final_results = final_results[:MAX_DOCS_FOR_LLM]
            diagnostics["docs_compressed"] = True
            diagnostics["original_doc_count"] = len(final_doc_ids)
        
        # Compress content length for LLM efficiency
        for result in final_results:
            if hasattr(result, 'content') and len(result.content) > MAX_CONTENT_LENGTH:
                result.content = result.content[:MAX_CONTENT_LENGTH] + "..."
                diagnostics.setdefault("content_compressed", 0)
                diagnostics["content_compressed"] += 1
        
        # Count generic penalties (documents with negative function scores)
        for result in final_results:
            section = result.metadata.get("section", "").lower()
            title = result.metadata.get("title", "").lower()
            if any(generic in section for generic in ["global", "overview", "platform"]) or \
               any(generic in title for generic in ["overview", "introduction", "welcome"]):
                diagnostics["generic_penalties"] += 1
        
        final_result = RetrievalResult(
            results=final_results,
            total_found=len(final_results),
            retrieval_time_ms=bm25_result.retrieval_time_ms + knn_result.retrieval_time_ms,
            method="enhanced_rrf",
            diagnostics=diagnostics
        )
        
        return final_result, diagnostics
        
    except Exception as e:
        logger.error(f"Enhanced RRF search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="enhanced_rrf",
            diagnostics=diagnostics
        ), diagnostics