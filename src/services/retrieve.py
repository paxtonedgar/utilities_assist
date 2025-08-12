"""Retrieval services for BM25, KNN, and RRF fusion."""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import SearchResult, RetrievalResult
from ..infra.opensearch_client import OpenSearchClient, SearchFilters

logger = logging.getLogger(__name__)


async def bm25_search(
    query: str,
    search_client: OpenSearchClient,
    index_name: str = "confluence_current",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    time_decay_days: int = 120
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
            index=index_name,
            k=top_k,
            time_decay_half_life_days=time_decay_days
        )
        
        # Convert to service format
        results = []
        for result in response.results:
            service_result = SearchResult(
                doc_id=result.doc_id,
                content=result.body,
                score=result.score,
                metadata={
                    "title": result.title,
                    **result.metadata
                }
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="bm25"
        )
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="bm25"
        )


async def knn_search(
    query_embedding: List[float],
    search_client: OpenSearchClient,
    index_name: str = "confluence_current",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    ef_search: int = 256
) -> RetrievalResult:
    """Perform KNN vector search with enterprise filters and HNSW optimization.
    
    Args:
        query_embedding: Query vector embedding (1536 dims)
        search_client: OpenSearch client with authentication
        index_name: OpenSearch index name or alias
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
            index=index_name,
            k=top_k,
            ef_search=ef_search
        )
        
        # Convert to service format
        results = []
        for result in response.results:
            service_result = SearchResult(
                doc_id=result.doc_id,
                content=result.body,
                score=result.score,
                metadata={
                    "title": result.title,
                    **result.metadata
                }
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=response.total_hits,
            retrieval_time_ms=response.took_ms,
            method="knn"
        )
        
    except Exception as e:
        logger.error(f"KNN search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="knn"
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
        # Convert service results back to client format for fusion
        from ..infra.opensearch_client import SearchResponse, SearchResult as ClientResult
        
        bm25_response = SearchResponse(
            results=[
                ClientResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    title=r.metadata.get("title", ""),
                    body=r.content,
                    metadata=r.metadata
                ) for r in bm25_result.results
            ],
            total_hits=bm25_result.total_found,
            took_ms=bm25_result.retrieval_time_ms,
            method="bm25"
        )
        
        knn_response = SearchResponse(
            results=[
                ClientResult(
                    doc_id=r.doc_id,
                    score=r.score,
                    title=r.metadata.get("title", ""),
                    body=r.content,
                    metadata=r.metadata
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
        
        # Convert back to service format
        results = []
        for result in fused_response.results:
            service_result = SearchResult(
                doc_id=result.doc_id,
                content=result.body,
                score=result.score,
                metadata={
                    "title": result.title,
                    **result.metadata
                }
            )
            results.append(service_result)
        
        return RetrievalResult(
            results=results,
            total_found=len(results),
            retrieval_time_ms=fused_response.took_ms,
            method="rrf"
        )
        
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=0,
            method="rrf"
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