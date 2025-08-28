# ARCHIVED: Old Hybrid Search Implementation (replaced by native OpenSearch hybrid search)
# Original location: src/infra/opensearch_client.py:590-665
# Replaced on: 2025-08-28
# Reason: OpenSearch native hybrid search with built-in RRF is more efficient

import logging
from typing import Optional, List
from src.infra.opensearch_client import SearchFilters, SearchResponse
from src.telemetry.logger import log_event, stage

logger = logging.getLogger(__name__)

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