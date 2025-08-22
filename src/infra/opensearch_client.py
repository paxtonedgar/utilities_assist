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

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import requests

from src.infra.settings import get_settings
from src.infra.search_config import OpenSearchConfig, QueryTemplates
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

    def __init__(self, settings: Optional[object] = None, extractor_cfg: Optional[ExtractorConfig] = None):
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
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias

        # Log search start
        log_event(
            stage="bm25",
            event="start",
            index=index,
            query_type="simple_match",
            k=k,
            filters_enabled=filters is not None,
            query_length=len(query),
        )

        # Build BM25 query matching working v1 branch structure
        search_body = self._build_simple_bm25_query(query, k, index)

        try:
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()

            # EXPLICIT OS QUERY LOG - for debugging query preservation
            logger.info(
                "OS_QUERY index=%s strategy=%s q=%r query_len=%d",
                index,
                "bm25",
                query,
                len(query),
            )

            # Use POST request for search with body
            _setup_jpmc_proxy()  # Ensure proxy is configured
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

            # Log HTTP response details
            log_event(
                stage="bm25",
                event="http_response",
                status=status_code,
                took_ms=took_ms,
                index=index,
            )

            # Handle HTTP errors
            if not response.ok:
                error_body = response.text[:500]  # Snippet only
                log_event(
                    stage="bm25",
                    event="error",
                    status=status_code,
                    took_ms=took_ms,
                    err=True,
                    error_type="HTTPError",
                    error_message=f"HTTP {status_code}",
                    error_body_snippet=error_body,
                    index=index,
                )
                response.raise_for_status()

            data = response.json()
            results = self._parse_search_response(data, index)

            # EXPLICIT OS RESPONSE LOG - for debugging query results
            top_result_title = results[0].title if results else None
            logger.info(
                "OS_RESPONSE index=%s took_ms=%.1f status=%s hits=%d top=%r",
                index,
                took_ms,
                status_code,
                len(results),
                top_result_title,
            )

            # Log successful completion
            total_hits = get_total_hits(data)
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

        except requests.exceptions.HTTPError as e:
            # Special handling for 404 - index not found
            if hasattr(e.response, "status_code") and e.response.status_code == 404:
                logger.warning(
                    f"Index not found: {index}. Continuing with empty results."
                )
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
                    status_code=e.response.status_code
                    if hasattr(e.response, "status_code")
                    else None,
                )

            return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")

        except Exception as e:
            # Log other errors
            log_event(
                stage="bm25",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                index=index,
                status_code=getattr(response, "status_code", None)
                if "response" in locals()
                else None,
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
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias

        # Log search start
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

        # Build simple kNN query (like main branch)
        search_body = self._build_simple_knn_query(query_vector, k, index)

        try:
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()

            # Use direct requests with auth (like main branch) instead of session
            _setup_jpmc_proxy()  # Ensure proxy is configured
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

            # Log HTTP response details
            log_event(
                stage="knn",
                event="http_response",
                status=status_code,
                took_ms=took_ms,
                index=index,
            )

            # Handle HTTP errors
            if not response.ok:
                error_body = response.text[:500]  # Snippet only
                log_event(
                    stage="knn",
                    event="error",
                    status=status_code,
                    took_ms=took_ms,
                    err=True,
                    error_type="HTTPError",
                    error_message=f"HTTP {status_code}",
                    error_body_snippet=error_body,
                    index=index,
                )
                response.raise_for_status()

            data = response.json()
            results = self._parse_search_response(data, index)

            # Log successful completion
            total_hits = get_total_hits(data)
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

        except requests.exceptions.HTTPError as e:
            # Special handling for 404 - index not found
            if hasattr(e.response, "status_code") and e.response.status_code == 404:
                logger.warning(
                    f"Index not found: {index}. Continuing with empty results."
                )
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
                    status_code=e.response.status_code
                    if hasattr(e.response, "status_code")
                    else None,
                )

            return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")

        except Exception as e:
            # Log other errors
            log_event(
                stage="knn",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                index=index,
                status_code=getattr(response, "status_code", None)
                if "response" in locals()
                else None,
            )

            return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")

    def rrf_fuse(
        self,
        bm25_response: SearchResponse,
        knn_response: SearchResponse,
        k: int = 8,
        rrf_k: int = 60,
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
        bm25_ranks = {
            result.doc_id: idx + 1 for idx, result in enumerate(bm25_response.results)
        }
        knn_ranks = {
            result.doc_id: idx + 1 for idx, result in enumerate(knn_response.results)
        }

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
                fused_result = Passage(
                    doc_id=result.doc_id,
                    index=result.index,
                    text=result.text,
                    section_title=result.section_title,
                    score=rrf_score,  # Use RRF-computed score
                    page_url=result.page_url,
                    api_name=result.api_name,
                    title=result.title,
                )
                fused_results.append(fused_result)

        logger.info(
            f"RRF fusion: {len(bm25_response.results)} BM25 + {len(knn_response.results)} kNN → {len(fused_results)} fused"
        )

        return SearchResponse(
            results=fused_results,
            total_hits=len(fused_results),
            took_ms=int((time.time() - start_time) * 1000)
            + bm25_response.took_ms
            + knn_response.took_ms,
            method="rrf",
        )

    @stage("hybrid")
    def hybrid_search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
        k: int = 50,
        time_decay_half_life_days: int = 120,
    ) -> SearchResponse:
        """
        Hybrid search using RRF fusion of separate BM25 and kNN searches.
        This approach avoids OpenSearch parsing issues with nested knn queries.

        Args:
            query: Search query string
            query_vector: Optional query embedding vector
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            time_decay_half_life_days: Half-life for time decay in days

        Returns:
            SearchResponse with hybrid search results
        """
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias

        # Log search start
        log_event(
            stage="hybrid",
            event="start",
            index=index,
            query_type="hybrid_rrf" if query_vector else "bm25_only",
            k=k,
            filters_enabled=filters is not None,
            query_length=len(query),
            has_vector=query_vector is not None,
        )

        # If no vector provided, fall back to BM25 search
        if not query_vector:
            logger.info("Hybrid search falling back to BM25 (no vector provided)")
            return self.bm25_search(query, filters, index, k, time_decay_half_life_days)

        try:
            # Perform separate BM25 and kNN searches
            logger.info("Performing separate BM25 and kNN searches for RRF fusion")

            # BM25 search
            bm25_response = self.bm25_search(
                query, filters, index, k, time_decay_half_life_days
            )

            # kNN search
            knn_response = self.knn_search(
                query_vector, filters, index, k, time_decay_half_life_days
            )

            # RRF fusion
            hybrid_response = self.rrf_fuse(bm25_response, knn_response, k=k, rrf_k=60)
            hybrid_response.method = "hybrid_rrf"

            logger.info(
                "Hybrid RRF fusion completed: BM25=%d hits, kNN=%d hits, fused=%d hits",
                bm25_response.total_hits,
                knn_response.total_hits,
                hybrid_response.total_hits,
            )

            return hybrid_response

        except Exception as e:
            logger.error(f"Hybrid search failed, falling back to BM25: {e}")
            return self.bm25_search(query, filters, index, k, time_decay_half_life_days)

    def _get_boosted_fields(self, index: Optional[str], search_type: str) -> List[str]:
        """Get boosted field list using centralized configuration."""
        index_name = index or self.settings.search_index_alias
        config = OpenSearchConfig._get_index_config(index_name)

        if search_type == "hybrid" or search_type == "mvrs":
            # MVRS boosting strategy: title^4, headings^2, body^1
            fields = []
            # Title fields with boost 4
            for field in config.title_fields:
                fields.append(f"{field}^4")
            # Section field as headings with boost 2
            fields.append("section^2")
            fields.append("section.text^2")  # Also search the analyzed version
            # Body field with boost 1 (default)
            fields.append("body^1")
            return fields
        elif search_type == "acronym":
            # Higher boosts for acronym queries
            fields = []
            for field in config.content_fields:
                fields.append(f"{field}^3")
            for field in config.title_fields:
                fields.append(f"{field}^8")  # Extra boost for titles in acronym queries
            # Add name field if available
            if "name" in [f.split("^")[0] for f in config.metadata_fields]:
                fields.append("name^4")
            return fields
        elif search_type == "bm25":
            # MVRS boosting strategy for BM25: title^4, headings^2, body^1
            fields = []
            for field in config.title_fields:
                fields.append(f"{field}^4")
            fields.append("section^2")
            fields.append("section.text^2")
            fields.append("body^1")
            return fields
        else:
            # Default field configuration
            return [f"{field}^2" for field in config.content_fields] + [
                f"{field}^3" for field in config.title_fields
            ]

    def _build_filter_clauses(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """Build filter clauses from SearchFilters."""
        clauses = []

        # ACL hash filter
        if filters.acl_hash:
            clauses.append({"term": {"acl_hash": filters.acl_hash}})

        # Space key filter
        if filters.space_key:
            clauses.append({"term": {"metadata.space_key": filters.space_key}})

        # Content type filter
        if filters.content_type:
            clauses.append({"term": {"content_type": filters.content_type}})

        # Time range filters - use consistent YYYY-MM-DD format
        if filters.updated_after or filters.updated_before:
            range_filter = {"range": {"updated_at": {}}}

            if filters.updated_after:
                range_filter["range"]["updated_at"]["gte"] = (
                    filters.updated_after.strftime("%Y-%m-%d")
                )

            if filters.updated_before:
                range_filter["range"]["updated_at"]["lte"] = (
                    filters.updated_before.strftime("%Y-%m-%d")
                )

            clauses.append(range_filter)

        return clauses

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
                passages=hit_passages
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

    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """Get the mapping (schema) for a specific index to discover field names."""
        try:
            url = f"{self.base_url}/{index_name}/_mapping"

            # Use direct requests with auth (like main branch)
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                logger.info("Using direct AWS4Auth for mapping request")
                response = requests.get(url, auth=aws_auth, timeout=30.0)
            else:
                logger.warning("No AWS auth available, using session for mapping")
                response = self.session.get(url, timeout=30.0)

            if not response.ok:
                error_body = response.text
                logger.error(
                    f"Get mapping failed with {response.status_code}: {error_body}"
                )
                return {}

            return response.json()

        except Exception as e:
            logger.error(f"Error getting index mapping for {index_name}: {e}")
            return {}

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
            health_url = f"{self.base_url}/_cluster/health"

            if self.settings.requires_aws_auth:
                # Use AWS auth for JPMC
                _setup_jpmc_proxy()  # Ensure proxy is configured
                aws_auth = _get_aws_auth()
                if aws_auth:
                    response = requests.get(health_url, auth=aws_auth, timeout=5.0)
                else:
                    # No auth available
                    raise ValueError("No authentication available for JPMC profile")
            else:
                response = self.session.get(health_url, timeout=5.0)

            response.raise_for_status()

            health = response.json()

            # Check if health data is valid
            if not health or not isinstance(health, dict):
                raise ValueError("Invalid health response format")

            # Test default index exists
            index_exists = self._check_index_exists(self.settings.search_index_alias)

            # Authentication status
            auth_status = "none"
            try:
                if (
                    self.session is not None
                    and hasattr(self.session, "auth")
                    and self.session.auth
                ):
                    auth_status = "session_configured"
                elif self.settings.requires_aws_auth:
                    auth_status = "aws_auth_configured"
            except:
                pass

            return {
                "status": "healthy",
                "cluster_name": health.get("cluster_name", "unknown"),
                "cluster_status": health.get("status", "unknown"),
                "node_count": health.get("number_of_nodes", 0),
                "data_nodes": health.get("number_of_data_nodes", 0),
                "index_exists": index_exists,
                "authentication": auth_status,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")

            # Safe authentication status for error case
            auth_status = "none"
            try:
                if (
                    self.session is not None
                    and hasattr(self.session, "auth")
                    and self.session.auth
                ):
                    auth_status = "session_configured"
                elif self.settings.requires_aws_auth:
                    auth_status = "aws_auth_configured"
            except:
                pass

            return {
                "status": "unhealthy",
                "error": str(e),
                "authentication": auth_status,
            }


def create_search_client(settings=None) -> OpenSearchClient:
    """Factory function to create OpenSearch client with centralized settings."""
    return OpenSearchClient(settings)
