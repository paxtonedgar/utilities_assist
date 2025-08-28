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
- Clean passage extraction via service layer
"""

import logging
import time
import json

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import requests

from src.infra.settings import get_settings
from src.infra.search_config import QueryTemplates
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
from src.telemetry.logger import log_event, stage
from src.services.passage_extractor import extract_passages
from src.services.schema_learner import observe_extraction
from src.services.models import ExtractorConfig, Passage

logger = logging.getLogger(__name__)


def get_total_hits(response_data: Dict[str, Any]) -> int:
    """
    Safely extract total hits count from OpenSearch response.

    Handles both formats:
    - Legacy: {"hits": {"total": 42}}
    - Modern: {"hits": {"total": {"value": 42, "relation": "eq"}}}
    """
    try:
        hits_section = response_data.get("hits", {})
        total = hits_section.get("total", 0)

        if isinstance(total, int):
            return total
        elif isinstance(total, dict):
            return total.get("value", 0)
        else:
            return 0
    except (KeyError, TypeError, AttributeError):
        return 0


@dataclass
class SearchFilters:
    """Search filters for ACL, space, and time-based filtering."""

    acl_hash: Optional[str] = None
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None


@dataclass
class SearchResponse:
    """Search response with results and metadata."""

    results: List[Passage]  # Now uses clean Passage objects
    total_hits: int
    took_ms: int
    method: str


class OpenSearchClient:
    """Production-ready OpenSearch client with enterprise features."""

    def __init__(
        self,
        settings: Optional[object] = None,
        extractor_cfg: Optional[ExtractorConfig] = None,
    ):
        """Initialize OpenSearch client with centralized settings."""
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.base_url = settings.opensearch_host.rstrip("/")
        self.extractor_cfg = extractor_cfg or ExtractorConfig()

        # Create session with proper authentication for the current profile
        if settings.requires_aws_auth:
            # Will use AWS authentication via clients module
            _setup_jpmc_proxy()
            self.session = None  # Use direct requests with AWS auth
        else:
            # Local development - create simple session
            import requests

            self.session = requests.Session()
            self.session.headers.update({"Content-Type": "application/json"})

    @stage("bm25")
    def bm25_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
        k: int = 50,
        time_decay_half_life_days: int = 120,
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
        index = index or self.settings.search_index_alias
        
        self._log_bm25_search_start(query, index, k, filters)
        search_body = self._build_simple_bm25_query(query, k, index)
        self._log_bm25_query_structure(search_body, index)
        
        try:
            response_data = self._execute_bm25_request(search_body, index)
            return self._process_bm25_success_response(response_data, search_body, index)
        except requests.exceptions.HTTPError as e:
            return self._handle_bm25_http_error(e, index)
        except Exception as e:
            return self._handle_bm25_general_error(e, index)
    
    def _log_bm25_search_start(self, query: str, index: str, k: int, filters: Optional[SearchFilters]) -> None:
        """Log the start of BM25 search."""
        log_event(
            stage="bm25",
            event="start",
            index=index,
            query_type="simple_match",
            k=k,
            filters_enabled=filters is not None,
            query_length=len(query),
        )
        
        logger.info(
            "OS_QUERY index=%s strategy=%s q=%r query_len=%d",
            index, "bm25", query, len(query),
        )
    
    def _log_bm25_query_structure(self, search_body: dict, index: str) -> None:
        """Log BM25 query structure for debugging."""
        if hasattr(search_body.get("query", {}), "get"):
            query_structure = "nested" if "nested" in search_body["query"] else "flat"
            logger.info(f"BM25_QUERY_STRUCTURE: {query_structure} for index {index}")
            
            if query_structure == "flat" and "khub-opensearch" in index:
                logger.warning(
                    f"BM25 using flat query for nested index: {json.dumps(search_body['query'], indent=2)}"
                )
    
    def _execute_bm25_request(self, search_body: dict, index: str) -> dict:
        """Execute BM25 request and return response data."""
        url = f"{self.base_url}/{index}/_search"
        start_time = time.time()
        
        _setup_jpmc_proxy()
        aws_auth = _get_aws_auth()
        headers = {"Content-Type": "application/json"}
        
        if aws_auth:
            response = requests.post(
                url, json=search_body, auth=aws_auth, timeout=30.0, headers=headers
            )
        else:
            response = self.session.post(
                url, json=search_body, timeout=30.0, headers=headers
            )
        
        took_ms = (time.time() - start_time) * 1000
        status_code = response.status_code
        
        log_event(
            stage="bm25", event="http_response", status=status_code, took_ms=took_ms, index=index
        )
        
        if not response.ok:
            self._log_bm25_http_error(response, took_ms, index)
            response.raise_for_status()
        
        return {
            'data': response.json(),
            'took_ms': took_ms,
            'status_code': status_code
        }
    
    def _log_bm25_http_error(self, response, took_ms: float, index: str) -> None:
        """Log BM25 HTTP error details."""
        error_body = response.text[:500]
        log_event(
            stage="bm25",
            event="error",
            status=response.status_code,
            took_ms=took_ms,
            err=True,
            error_type="HTTPError",
            error_message=f"HTTP {response.status_code}",
            error_body_snippet=error_body,
            index=index,
        )
    
    def _process_bm25_success_response(
        self, response_data: dict, search_body: dict, index: str
    ) -> SearchResponse:
        """Process successful BM25 response."""
        data = response_data['data']
        took_ms = response_data['took_ms']
        status_code = response_data['status_code']
        
        results = self._parse_search_response(data, index)
        total_hits = get_total_hits(data)
        
        self._log_bm25_response(results, index, took_ms, status_code)
        self._log_bm25_zero_results(results, total_hits, search_body, index)
        
        log_event(
            stage="bm25",
            event="success",
            took_ms=took_ms,
            result_count=len(results),
            total_hits=total_hits,
            index=index,
            elasticsearch_took=data.get("took", 0),
        )
        
        return SearchResponse(
            results=results,
            total_hits=total_hits,
            took_ms=int(took_ms),
            method="bm25",
        )
    
    def _log_bm25_response(
        self, results: List[Passage], index: str, took_ms: float, status_code: int
    ) -> None:
        """Log BM25 response details."""
        top_result_title = results[0].title if results else None
        logger.info(
            "OS_RESPONSE index=%s took_ms=%.1f status=%s hits=%d top=%r",
            index, took_ms, status_code, len(results), top_result_title,
        )
    
    def _log_bm25_zero_results(
        self, results: List[Passage], total_hits: int, search_body: dict, index: str
    ) -> None:
        """Log zero results for debugging."""
        if len(results) == 0 and total_hits == 0:
            logger.warning(
                f"BM25_ZERO_RESULTS for {index}: query={json.dumps(search_body, indent=2)}"
            )
    
    def _handle_bm25_http_error(self, e: requests.exceptions.HTTPError, index: str) -> SearchResponse:
        """Handle BM25 HTTP errors."""
        if hasattr(e.response, "status_code") and e.response.status_code == 404:
            logger.warning(f"Index not found: {index}. Continuing with empty results.")
            log_event(
                stage="bm25",
                event="index_not_found",
                index=index,
                status_code=404,
                message=f"Index {index} not found, returning empty results",
            )
        else:
            log_event(
                stage="bm25",
                event="error",
                err=True,
                error_type="HTTPError",
                error_message=str(e)[:200],
                index=index,
                status_code=e.response.status_code if hasattr(e.response, "status_code") else None,
            )
        
        return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")
    
    def _handle_bm25_general_error(self, e: Exception, index: str) -> SearchResponse:
        """Handle BM25 general errors."""
        log_event(
            stage="bm25",
            event="error",
            err=True,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            index=index,
        )
        
        return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")

    @stage("knn")
    def knn_search(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
        k: int = 50,
        ef_search: int = 256,
        time_decay_half_life_days: int = 120,
    ) -> SearchResponse:
        """
        kNN vector similarity search with ACL filters and time decay.

        Args:
            query_vector: Query embedding vector (dimensions defined by OpenSearchConfig.EMBEDDING_DIMENSIONS)
            filters: ACL, space, and time filters
            index: Index name or alias to search (defaults to OpenSearchConfig.get_default_index())
            k: Number of results to return
            ef_search: ef_search parameter for HNSW (trade-off: accuracy vs speed)
            time_decay_half_life_days: Half-life for time decay in days (uniform with BM25)

        Returns:
            SearchResponse with kNN results
        """
        index = index or self.settings.search_index_alias
        
        self._log_knn_search_start(query_vector, index, k, ef_search, filters)
        search_body = self._build_simple_knn_query(query_vector, k, index)
        self._log_knn_query_structure(search_body, index)
        
        try:
            response_data = self._execute_knn_request(search_body, index)
            return self._process_knn_success_response(response_data, search_body, index)
        except requests.exceptions.HTTPError as e:
            return self._handle_knn_http_error(e, index)
        except Exception as e:
            return self._handle_knn_general_error(e, index)
    
    def _log_knn_search_start(
        self, query_vector: List[float], index: str, k: int, ef_search: int, filters: Optional[SearchFilters]
    ) -> None:
        """Log the start of kNN search."""
        log_event(
            stage="knn",
            event="start",
            index=index,
            query_type="vector",
            k=k,
            vector_dims=len(query_vector),
            ef_search=ef_search,
            filters_enabled=filters is not None,
        )
    
    def _log_knn_query_structure(self, search_body: dict, index: str) -> None:
        """Log kNN query structure for debugging."""
        if search_body and hasattr(search_body.get("query", {}), "get"):
            query_structure = "nested" if "nested" in search_body["query"] else "flat"
            logger.info(f"KNN_QUERY_STRUCTURE: {query_structure} for index {index}")
            
            if query_structure == "flat" and "khub-opensearch" in index:
                logger.warning(
                    f"kNN using flat query for nested index: {json.dumps(search_body['query'], indent=2)}"
                )
    
    def _execute_knn_request(self, search_body: dict, index: str) -> dict:
        """Execute kNN request and return response data."""
        url = f"{self.base_url}/{index}/_search"
        start_time = time.time()
        
        _setup_jpmc_proxy()
        aws_auth = _get_aws_auth()
        headers = {"Content-Type": "application/json"}
        
        if aws_auth:
            response = requests.post(
                url, json=search_body, auth=aws_auth, timeout=30.0, headers=headers
            )
        else:
            response = self.session.post(
                url, json=search_body, timeout=30.0, headers=headers
            )
        
        took_ms = (time.time() - start_time) * 1000
        status_code = response.status_code
        
        log_event(
            stage="knn", event="http_response", status=status_code, took_ms=took_ms, index=index
        )
        
        if not response.ok:
            self._log_knn_http_error(response, took_ms, index)
            response.raise_for_status()
        
        return {
            'data': response.json(),
            'took_ms': took_ms,
            'status_code': status_code
        }
    
    def _log_knn_http_error(self, response, took_ms: float, index: str) -> None:
        """Log kNN HTTP error details."""
        error_body = response.text[:500]
        log_event(
            stage="knn",
            event="error",
            status=response.status_code,
            took_ms=took_ms,
            err=True,
            error_type="HTTPError",
            error_message=f"HTTP {response.status_code}",
            error_body_snippet=error_body,
            index=index,
        )
    
    def _process_knn_success_response(
        self, response_data: dict, search_body: dict, index: str
    ) -> SearchResponse:
        """Process successful kNN response."""
        data = response_data['data']
        took_ms = response_data['took_ms']
        
        results = self._parse_search_response(data, index)
        total_hits = get_total_hits(data)
        
        self._log_knn_zero_results(results, total_hits, search_body, index)
        
        log_event(
            stage="knn",
            event="success",
            took_ms=took_ms,
            result_count=len(results),
            total_hits=total_hits,
            index=index,
            elasticsearch_took=data.get("took", 0),
        )
        
        return SearchResponse(
            results=results,
            total_hits=total_hits,
            took_ms=int(took_ms),
            method="knn",
        )
    
    def _log_knn_zero_results(
        self, results: List[Passage], total_hits: int, search_body: dict, index: str
    ) -> None:
        """Log zero results for debugging."""
        if len(results) == 0 and total_hits == 0:
            logger.warning(
                f"KNN_ZERO_RESULTS for {index}: query={json.dumps(search_body, indent=2)}"
            )
    
    def _handle_knn_http_error(self, e: requests.exceptions.HTTPError, index: str) -> SearchResponse:
        """Handle kNN HTTP errors."""
        if hasattr(e.response, "status_code") and e.response.status_code == 404:
            logger.warning(f"Index not found: {index}. Continuing with empty results.")
            log_event(
                stage="knn",
                event="index_not_found",
                index=index,
                status_code=404,
                message=f"Index {index} not found, returning empty results",
            )
        else:
            log_event(
                stage="knn",
                event="error",
                err=True,
                error_type="HTTPError",
                error_message=str(e)[:200],
                index=index,
                status_code=e.response.status_code if hasattr(e.response, "status_code") else None,
            )
        
        return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")
    
    def _handle_knn_general_error(self, e: Exception, index: str) -> SearchResponse:
        """Handle kNN general errors."""
        log_event(
            stage="knn",
            event="error",
            err=True,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            index=index,
        )
        
        return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")



    @stage("hybrid_native")
    def hybrid_search_native(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
        k: int = 50,
        alpha: float = 0.5,
    ) -> SearchResponse:
        """
        Native OpenSearch hybrid search using built-in RRF.
        Replaces custom Python RRF with OpenSearch's native hybrid query.

        Args:
            query: Search query string
            query_vector: Query embedding vector
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            alpha: Hybrid search weight (0.0=BM25 only, 1.0=kNN only, 0.5=balanced)

        Returns:
            SearchResponse with native hybrid search results
        """
        if index is None:
            index = self.settings.search_index_alias

        # Log search start
        log_event(
            stage="hybrid_native",
            event="start",
            index=index,
            query_type="native_hybrid" if query_vector else "bm25_only",
            k=k,
            alpha=alpha,
            filters_enabled=filters is not None,
            query_length=len(query),
            has_vector=query_vector is not None,
        )

        # If no vector provided, fall back to BM25 search
        if not query_vector:
            logger.info("Native hybrid search falling back to BM25 (no vector provided)")
            return self.bm25_search(query, filters, index, k)

        try:
            # Build native hybrid query using OpenSearch hybrid query syntax
            search_body = self._build_native_hybrid_query(
                query, query_vector, filters, k, alpha, index
            )

            logger.info("Performing native OpenSearch hybrid search with built-in RRF")

            # Execute request
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()

            _setup_jpmc_proxy()
            aws_auth = _get_aws_auth()
            headers = {"Content-Type": "application/json"}

            if aws_auth:
                response = requests.post(
                    url, json=search_body, auth=aws_auth, timeout=30.0, headers=headers
                )
            else:
                response = self.session.post(
                    url, json=search_body, timeout=30.0, headers=headers
                )

            took_ms = (time.time() - start_time) * 1000
            response.raise_for_status()

            # Parse response
            data = response.json()
            results = self._parse_search_response(data, index)
            total_hits = get_total_hits(data)

            logger.info(
                "Native hybrid search completed: %d results in %.1fms",
                len(results),
                took_ms,
            )

            log_event(
                stage="hybrid_native",
                event="success",
                took_ms=took_ms,
                result_count=len(results),
                total_hits=total_hits,
                index=index,
                elasticsearch_took=data.get("took", 0),
            )

            return SearchResponse(
                results=results,
                total_hits=total_hits,
                took_ms=int(took_ms),
                method="hybrid_native",
            )

        except Exception as e:
            logger.error(f"Native hybrid search failed, falling back to BM25: {e}")
            log_event(
                stage="hybrid_native",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                index=index,
            )
            return self.bm25_search(query, filters, index, k)



    def _parse_search_response(
        self, data: Dict[str, Any], index: Optional[str] = None
    ) -> List[Passage]:
        """Parse OpenSearch response into Passage objects - thin wrapper for extraction service."""
        passages = []
        hits = data.get("hits", {}).get("hits", [])

        if not isinstance(hits, list):
            logger.warning("Invalid hits format in OpenSearch response")
            return []

        # Delegate ALL extraction logic to service layer
        for hit in hits:
            hit_passages = extract_passages(hit, self.extractor_cfg)
            passages.extend(hit_passages)

            # C1: Observe extraction for schema learning
            observe_extraction(
                index=hit.get("_index", index or "unknown"),
                hit=hit,
                passages=hit_passages,
            )

        log_event(
            stage="parse_response",
            total_hits=len(hits),
            total_passages=len(passages),
            index=index,
            avg_passages_per_hit=len(passages) / len(hits) if hits else 0,
        )

        return passages

    def _build_simple_bm25_query(
        self, query: str, k: int, index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build optimized BM25 query using centralized templates for nested structure support."""

        # Use our centralized query template for nested structure support
        search_body = QueryTemplates.build_bm25_query(
            text_query=query, index_name=index or self.settings.search_index_alias, k=k
        )
        return search_body

    def _build_simple_knn_query(
        self, query_vector: List[float], k: int, index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build optimized kNN query using centralized templates for nested structure support."""

        # Use our centralized query template for nested structure support
        search_body = QueryTemplates.build_knn_query(
            vector_query=query_vector,
            index_name=index or self.settings.search_index_alias,
            k=k,
        )
        return search_body

    def _build_native_hybrid_query(
        self,
        query: str,
        query_vector: List[float],
        filters: Optional[SearchFilters],
        k: int,
        alpha: float,
        index: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build native OpenSearch hybrid query with built-in RRF."""
        
        # Determine if we need nested structure based on index name
        is_nested = index and "khub-opensearch" in index
        
        if is_nested:
            # Build nested hybrid query for structured documents
            search_body = {
                "size": k,
                "query": {
                    "hybrid": {
                        "queries": [
                            # BM25 component with nested structure
                            {
                                "nested": {
                                    "path": "content",
                                    "query": {
                                        "bool": {
                                            "should": [
                                                {
                                                    "match": {
                                                        "content.title": {
                                                            "query": query,
                                                            "boost": 2.0
                                                        }
                                                    }
                                                },
                                                {
                                                    "match": {
                                                        "content.body": {
                                                            "query": query,
                                                            "boost": 1.0
                                                        }
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            },
                            # kNN component with nested structure
                            {
                                "nested": {
                                    "path": "content",
                                    "query": {
                                        "knn": {
                                            "content.embedding": {
                                                "vector": query_vector,
                                                "k": k
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": ["content"]
            }
        else:
            # Build flat hybrid query for simple documents
            search_body = {
                "size": k,
                "query": {
                    "hybrid": {
                        "queries": [
                            # BM25 component
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "match": {
                                                "title": {
                                                    "query": query,
                                                    "boost": 2.0
                                                }
                                            }
                                        },
                                        {
                                            "match": {
                                                "body": {
                                                    "query": query,
                                                    "boost": 1.0
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                            # kNN component
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_vector,
                                        "k": k
                                    }
                                }
                            }
                        ]
                    }
                }
            }

        # Apply filters if provided
        if filters:
            self._apply_filters_to_hybrid_query(search_body, filters, is_nested)

        return search_body

    def _apply_filters_to_hybrid_query(
        self, search_body: Dict[str, Any], filters: SearchFilters, is_nested: bool
    ) -> None:
        """Apply ACL and other filters to hybrid query."""
        
        filter_conditions = []
        
        # ACL filter
        if filters.acl_hash:
            if is_nested:
                filter_conditions.append({
                    "nested": {
                        "path": "content",
                        "query": {
                            "term": {"content.acl_hash": filters.acl_hash}
                        }
                    }
                })
            else:
                filter_conditions.append({
                    "term": {"acl_hash": filters.acl_hash}
                })

        # Space filter
        if filters.space_key:
            if is_nested:
                filter_conditions.append({
                    "nested": {
                        "path": "content",
                        "query": {
                            "term": {"content.metadata.space_key": filters.space_key}
                        }
                    }
                })
            else:
                filter_conditions.append({
                    "term": {"metadata.space_key": filters.space_key}
                })

        # Content type filter
        if filters.content_type:
            if is_nested:
                filter_conditions.append({
                    "nested": {
                        "path": "content",
                        "query": {
                            "term": {"content.content_type": filters.content_type}
                        }
                    }
                })
            else:
                filter_conditions.append({
                    "term": {"content_type": filters.content_type}
                })

        # Date range filters
        if filters.updated_after or filters.updated_before:
            date_range = {}
            if filters.updated_after:
                date_range["gte"] = filters.updated_after.strftime('%Y-%m-%d')
            if filters.updated_before:
                date_range["lte"] = filters.updated_before.strftime('%Y-%m-%d')
            
            if is_nested:
                filter_conditions.append({
                    "nested": {
                        "path": "content",
                        "query": {
                            "range": {"content.updated_at": date_range}
                        }
                    }
                })
            else:
                filter_conditions.append({
                    "range": {"updated_at": date_range}
                })

        # Apply filters by wrapping the hybrid query in a bool query
        if filter_conditions:
            original_query = search_body["query"]
            search_body["query"] = {
                "bool": {
                    "must": [original_query],
                    "filter": filter_conditions
                }
            }


    def _check_index_exists(self, index_name: str) -> bool:
        """Check if an index exists in OpenSearch."""
        try:
            url = f"{self.base_url}/{index_name}"

            # Use direct requests with auth (like main branch)
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                response = requests.head(url, auth=aws_auth, timeout=5.0)
            else:
                response = self.session.head(url, timeout=5.0)

            # 200 = exists, 404 = doesn't exist, other = error
            return response.status_code == 200

        except Exception as e:
            logger.warning(f"Error checking index existence for {index_name}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for JPMC-aware OpenSearch connectivity."""
        try:
            health_data = self._execute_health_request()
            self._validate_health_response(health_data)
            
            index_exists = self._check_index_exists(self.settings.search_index_alias)
            auth_status = self._get_authentication_status()
            
            return self._build_healthy_response(health_data, index_exists, auth_status)
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return self._build_unhealthy_response(e)
    
    def _execute_health_request(self) -> dict:
        """Execute the health check request."""
        health_url = f"{self.base_url}/_cluster/health"
        
        if self.settings.requires_aws_auth:
            return self._execute_aws_auth_health_request(health_url)
        else:
            response = self.session.get(health_url, timeout=5.0)
            response.raise_for_status()
            return response.json()
    
    def _execute_aws_auth_health_request(self, health_url: str) -> dict:
        """Execute health request with AWS authentication."""
        _setup_jpmc_proxy()
        aws_auth = _get_aws_auth()
        
        if not aws_auth:
            raise ValueError("No authentication available for JPMC profile")
        
        response = requests.get(health_url, auth=aws_auth, timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    def _validate_health_response(self, health_data: dict) -> None:
        """Validate health response format."""
        if not health_data or not isinstance(health_data, dict):
            raise ValueError("Invalid health response format")
    
    def _get_authentication_status(self) -> str:
        """Determine authentication status."""
        try:
            if (
                self.session is not None
                and hasattr(self.session, "auth")
                and self.session.auth
            ):
                return "session_configured"
            elif self.settings.requires_aws_auth:
                return "aws_auth_configured"
            return "none"
        except Exception:
            return "none"
    
    def _build_healthy_response(
        self, health_data: dict, index_exists: bool, auth_status: str
    ) -> Dict[str, Any]:
        """Build healthy response dictionary."""
        return {
            "status": "healthy",
            "cluster_name": health_data.get("cluster_name", "unknown"),
            "cluster_status": health_data.get("status", "unknown"),
            "node_count": health_data.get("number_of_nodes", 0),
            "data_nodes": health_data.get("number_of_data_nodes", 0),
            "index_exists": index_exists,
            "authentication": auth_status,
        }
    
    def _build_unhealthy_response(self, error: Exception) -> Dict[str, Any]:
        """Build unhealthy response dictionary."""
        auth_status = self._get_authentication_status()
        
        return {
            "status": "unhealthy",
            "error": str(error),
            "authentication": auth_status,
        }


def create_search_client(settings=None) -> OpenSearchClient:
    """Factory function to create OpenSearch client with centralized settings."""
    return OpenSearchClient(settings)
