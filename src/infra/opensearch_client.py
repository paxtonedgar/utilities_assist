# src/infra/opensearch_client.py
"""
OpenSearch client with enterprise-grade search capabilities.

All filters must be applied to both BM25 and kNN to avoid leakage.

Features:
- Pooled session with JPMC authentication and proxy support
- BM25 search with ACL filters and time decay
- kNN vector search with ef_search optimization
- Pure Python RRF fusion for hybrid search
- Comprehensive filtering (ACL, space, time ranges)
- Function score time decay for recency boosting
"""

import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import requests

from .config import SearchCfg
from .clients import make_search_session

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Search filters for ACL, space, and time-based filtering."""
    acl_hash: Optional[str] = None
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None


@dataclass 
class SearchResult:
    """Individual search result with score and metadata."""
    doc_id: str
    score: float
    title: str
    body: str
    metadata: Dict[str, Any]
    
    
@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[SearchResult]
    total_hits: int
    took_ms: int
    method: str


class OpenSearchClient:
    """Production-ready OpenSearch client with enterprise features."""
    
    def __init__(self, config: SearchCfg):
        self.config = config
        self.session = make_search_session(config)
        self.base_url = config.host.rstrip('/')
        
    def bm25_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        index: str = "confluence_current",
        k: int = 50,
        time_decay_half_life_days: int = 120
    ) -> SearchResponse:
        """
        BM25 full-text search with ACL filters and time decay.
        
        Args:
            query: Search query string
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            time_decay_half_life_days: Half-life for time decay in days
            
        Returns:
            SearchResponse with BM25 results
        """
        start_time = time.time()
        
        # Build BM25 query with filters
        search_body = self._build_bm25_query(
            query, filters, k, time_decay_half_life_days
        )
        
        try:
            url = f"{self.base_url}/{index}/_search"
            response = self.session.post(url, json=search_body, timeout=self.config.timeout_s)
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            return SearchResponse(
                results=results,
                total_hits=data["hits"]["total"]["value"],
                took_ms=int((time.time() - start_time) * 1000),
                method="bm25"
            )
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")
    
    def knn_search(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters] = None,
        index: str = "confluence_current", 
        k: int = 50,
        ef_search: int = 256
    ) -> SearchResponse:
        """
        kNN vector similarity search with ACL filters.
        
        Args:
            query_vector: Query embedding vector (1536 dims)
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            ef_search: ef_search parameter for HNSW (trade-off: accuracy vs speed)
            
        Returns:
            SearchResponse with kNN results
        """
        start_time = time.time()
        
        # Build kNN query with filters
        search_body = self._build_knn_query(query_vector, filters, k, ef_search)
        
        try:
            url = f"{self.base_url}/{index}/_search"
            response = self.session.post(url, json=search_body, timeout=self.config.timeout_s)
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            return SearchResponse(
                results=results,
                total_hits=data["hits"]["total"]["value"],
                took_ms=int((time.time() - start_time) * 1000),
                method="knn"
            )
            
        except Exception as e:
            logger.error(f"kNN search failed: {e}")
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")
    
    def rrf_fuse(
        self,
        bm25_response: SearchResponse,
        knn_response: SearchResponse,
        k: int = 8,
        rrf_k: int = 60
    ) -> SearchResponse:
        """
        Reciprocal Rank Fusion (RRF) for hybrid search.
        
        Pure Python implementation combining BM25 and kNN results.
        
        Args:
            bm25_response: BM25 search results
            knn_response: kNN search results
            k: Final number of results to return
            rrf_k: RRF constant (typically 60, higher = less aggressive fusion)
            
        Returns:
            SearchResponse with fused results
        """
        start_time = time.time()
        
        # Create rank maps for both result sets
        bm25_ranks = {result.doc_id: idx + 1 for idx, result in enumerate(bm25_response.results)}
        knn_ranks = {result.doc_id: idx + 1 for idx, result in enumerate(knn_response.results)}
        
        # Create document map for result data
        doc_map = {}
        for result in bm25_response.results:
            doc_map[result.doc_id] = result
        for result in knn_response.results:
            doc_map[result.doc_id] = result
        
        # Calculate RRF scores
        rrf_scores = {}
        all_doc_ids = set(bm25_ranks.keys()) | set(knn_ranks.keys())
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Add BM25 contribution: 1 / (k + rank)
            if doc_id in bm25_ranks:
                rrf_score += 1.0 / (rrf_k + bm25_ranks[doc_id])
            
            # Add kNN contribution: 1 / (k + rank)
            if doc_id in knn_ranks:
                rrf_score += 1.0 / (rrf_k + knn_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score (descending) and take top k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build final results with RRF scores
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_map:
                result = doc_map[doc_id]
                # Create new result with RRF score
                fused_result = SearchResult(
                    doc_id=result.doc_id,
                    score=rrf_score,
                    title=result.title,
                    body=result.body,
                    metadata=result.metadata
                )
                fused_results.append(fused_result)
        
        logger.info(f"RRF fusion: {len(bm25_response.results)} BM25 + {len(knn_response.results)} kNN → {len(fused_results)} fused")
        
        return SearchResponse(
            results=fused_results,
            total_hits=len(fused_results),
            took_ms=int((time.time() - start_time) * 1000) + bm25_response.took_ms + knn_response.took_ms,
            method="rrf"
        )
    
    def _build_bm25_query(
        self,
        query: str,
        filters: Optional[SearchFilters],
        k: int,
        time_decay_half_life_days: int
    ) -> Dict[str, Any]:
        """Build BM25 query with filters and time decay."""
        
        # Base multi-match query
        base_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^2.0", "body^1.0"],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "max_expansions": 50
            }
        }
        
        # Apply filters to query
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
            if filter_clauses:
                base_query = {
                    "bool": {
                        "must": [base_query],
                        "filter": filter_clauses
                    }
                }
        
        # Apply time decay via function_score
        if time_decay_half_life_days > 0:
            decay_query = {
                "function_score": {
                    "query": base_query,
                    "functions": [
                        {
                            "exp": {
                                "updated_at": {
                                    "scale": f"{time_decay_half_life_days}d",
                                    "offset": "0d",
                                    "decay": 0.5
                                }
                            },
                            "weight": 0.3  # 30% weight for recency
                        }
                    ],
                    "score_mode": "multiply",
                    "boost_mode": "multiply"
                }
            }
            final_query = decay_query
        else:
            final_query = base_query
        
        return {
            "query": final_query,
            "size": k,
            "sort": [{"_score": {"order": "desc"}}],
            "_source": ["title", "body", "metadata", "updated_at", "page_id", "canonical_id"]
        }
    
    def _build_knn_query(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters],
        k: int,
        ef_search: int
    ) -> Dict[str, Any]:
        """Build kNN query with filters."""
        
        # Base kNN query
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                    "ef_search": ef_search
                }
            }
        }
        
        # Apply filters to query
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
            if filter_clauses:
                knn_query = {
                    "bool": {
                        "must": [knn_query],
                        "filter": filter_clauses
                    }
                }
        
        return {
            "query": knn_query,
            "size": k,
            "sort": [{"_score": {"order": "desc"}}],
            "_source": ["title", "body", "metadata", "updated_at", "page_id", "canonical_id"]
        }
    
    def _build_filter_clauses(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """Build filter clauses from SearchFilters."""
        clauses = []
        
        # ACL hash filter
        if filters.acl_hash:
            clauses.append({
                "term": {"acl_hash": filters.acl_hash}
            })
        
        # Space key filter
        if filters.space_key:
            clauses.append({
                "term": {"metadata.space_key": filters.space_key}
            })
        
        # Content type filter
        if filters.content_type:
            clauses.append({
                "term": {"content_type": filters.content_type}
            })
        
        # Time range filters
        if filters.updated_after or filters.updated_before:
            range_filter = {"range": {"updated_at": {}}}
            
            if filters.updated_after:
                range_filter["range"]["updated_at"]["gte"] = filters.updated_after.isoformat()
            
            if filters.updated_before:
                range_filter["range"]["updated_at"]["lte"] = filters.updated_before.isoformat()
            
            clauses.append(range_filter)
        
        return clauses
    
    def _parse_search_response(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse OpenSearch response into SearchResult objects."""
        results = []
        
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            
            result = SearchResult(
                doc_id=hit["_id"],
                score=hit["_score"],
                title=source.get("title", ""),
                body=source.get("body", ""),
                metadata=source.get("metadata", {})
            )
            results.append(result)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch cluster health and connectivity."""
        try:
            # Test basic connectivity
            url = f"{self.base_url}/_cluster/health"
            response = self.session.get(url, timeout=5.0)
            response.raise_for_status()
            
            health = response.json()
            
            # Test index existence
            alias_url = f"{self.base_url}/confluence_current"
            alias_response = self.session.head(alias_url, timeout=5.0)
            index_exists = alias_response.status_code == 200
            
            return {
                "status": "healthy" if health["status"] in ["green", "yellow"] else "unhealthy",
                "cluster_name": health["cluster_name"],
                "cluster_status": health["status"],
                "node_count": health["number_of_nodes"],
                "data_nodes": health["number_of_data_nodes"],
                "index_exists": index_exists,
                "authentication": "configured" if self.session.auth else "none"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "authentication": "configured" if self.session.auth else "none"
            }


def create_search_client(config: SearchCfg) -> OpenSearchClient:
    """Factory function to create OpenSearch client with proper configuration."""
    return OpenSearchClient(config)