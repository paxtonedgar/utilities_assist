"""Retrieval services for BM25, KNN, and RRF fusion."""

import logging
import time
import json
from typing import List, Dict, Any, Optional
import requests
from .models import SearchResult, RetrievalResult

logger = logging.getLogger(__name__)


async def bm25_search(
    query: str,
    session: requests.Session,
    index_name: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> RetrievalResult:
    """Perform BM25 text search using OpenSearch.
    
    Args:
        query: Search query text
        session: HTTP session for OpenSearch
        index_name: OpenSearch index name
        top_k: Number of results to return
        filters: Additional filters to apply
        
    Returns:
        RetrievalResult with BM25 search results
    """
    start_time = time.time()
    
    try:
        # Build OpenSearch query
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "title^3", "summary"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "size": top_k,
            "sort": ["_score"],
            "_source": ["content", "title", "doc_id", "metadata"]
        }
        
        # Apply filters if provided
        if filters:
            search_body["query"]["bool"]["filter"] = _build_filters(filters)
        
        # Execute search
        search_url = f"/{index_name}/_search"
        response = session.post(search_url, json=search_body)
        response.raise_for_status()
        
        results = response.json()
        search_results = _parse_opensearch_results(results, "bm25")
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            results=search_results,
            total_found=results.get("hits", {}).get("total", {}).get("value", 0),
            retrieval_time_ms=retrieval_time,
            method="bm25"
        )
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            method="bm25"
        )


async def knn_search(
    query_embedding: List[float],
    session: requests.Session,
    index_name: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> RetrievalResult:
    """Perform KNN vector search using OpenSearch.
    
    Args:
        query_embedding: Query vector embedding
        session: HTTP session for OpenSearch
        index_name: OpenSearch index name
        top_k: Number of results to return
        filters: Additional filters to apply
        
    Returns:
        RetrievalResult with KNN search results
    """
    start_time = time.time()
    
    try:
        # Build KNN query
        search_body = {
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "size": top_k,
            "_source": ["content", "title", "doc_id", "metadata"]
        }
        
        # Apply filters if provided
        if filters:
            # For KNN, filters are applied differently
            search_body["query"] = {
                "bool": {
                    "must": [search_body["query"]],
                    "filter": _build_filters(filters)
                }
            }
        
        # Execute search
        search_url = f"/{index_name}/_search"
        response = session.post(search_url, json=search_body)
        response.raise_for_status()
        
        results = response.json()
        search_results = _parse_opensearch_results(results, "knn")
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            results=search_results,
            total_found=len(search_results),
            retrieval_time_ms=retrieval_time,
            method="knn"
        )
        
    except Exception as e:
        logger.error(f"KNN search failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            method="knn"
        )


async def rrf_fuse(
    bm25_result: RetrievalResult,
    knn_result: RetrievalResult,
    rrf_constant: int = 60,
    top_k: int = 10
) -> RetrievalResult:
    """Fuse BM25 and KNN results using Reciprocal Rank Fusion.
    
    Args:
        bm25_result: Results from BM25 search
        knn_result: Results from KNN search
        rrf_constant: RRF constant (typically 60)
        top_k: Number of final results to return
        
    Returns:
        RetrievalResult with fused and re-ranked results
    """
    start_time = time.time()
    
    try:
        # Create document score maps
        doc_scores = {}
        doc_results = {}
        
        # Process BM25 results
        for rank, result in enumerate(bm25_result.results):
            rrf_score = 1.0 / (rrf_constant + rank + 1)
            doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0) + rrf_score
            doc_results[result.doc_id] = result
        
        # Process KNN results
        for rank, result in enumerate(knn_result.results):
            rrf_score = 1.0 / (rrf_constant + rank + 1)
            doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0) + rrf_score
            if result.doc_id not in doc_results:
                doc_results[result.doc_id] = result
        
        # Sort by RRF score and take top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build final results
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            result = doc_results[doc_id]
            # Update score to RRF score
            fused_result = SearchResult(
                doc_id=result.doc_id,
                score=rrf_score,
                content=result.content,
                metadata=result.metadata
            )
            fused_results.append(fused_result)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            results=fused_results,
            total_found=len(fused_results),
            retrieval_time_ms=retrieval_time,
            method="rrf"
        )
        
    except Exception as e:
        logger.error(f"RRF fusion failed: {e}")
        return RetrievalResult(
            results=[],
            total_found=0,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            method="rrf"
        )


def _build_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build OpenSearch filter clauses."""
    filter_clauses = []
    
    for field, value in filters.items():
        if isinstance(value, list):
            # Terms filter for multiple values
            filter_clauses.append({
                "terms": {field: value}
            })
        elif isinstance(value, str):
            # Term filter for single value
            filter_clauses.append({
                "term": {f"{field}.keyword": value}
            })
        elif isinstance(value, dict) and "range" in value:
            # Range filter
            filter_clauses.append({
                "range": {field: value["range"]}
            })
    
    return filter_clauses


def _parse_opensearch_results(opensearch_response: Dict[str, Any], method: str) -> List[SearchResult]:
    """Parse OpenSearch response into SearchResult objects."""
    results = []
    
    try:
        hits = opensearch_response.get("hits", {}).get("hits", [])
        
        for hit in hits:
            source = hit.get("_source", {})
            
            result = SearchResult(
                doc_id=source.get("doc_id", hit.get("_id", "")),
                score=float(hit.get("_score", 0.0)),
                content=source.get("content", ""),
                metadata={
                    **source.get("metadata", {}),
                    "title": source.get("title", ""),
                    "method": method
                }
            )
            results.append(result)
            
    except Exception as e:
        logger.error(f"Failed to parse OpenSearch results: {e}")
    
    return results