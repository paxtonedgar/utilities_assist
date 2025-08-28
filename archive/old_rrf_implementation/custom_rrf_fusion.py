# ARCHIVED: Custom RRF Fusion Implementation (replaced by native OpenSearch hybrid search)
# Original location: src/infra/opensearch_client.py:499-588
# Replaced on: 2025-08-28
# Reason: OpenSearch native hybrid search with built-in RRF is more efficient

import time
import logging
from typing import Dict, List, Any, Optional
from src.services.models import Passage, SearchResponse

logger = logging.getLogger(__name__)

def rrf_fuse(
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
            # Create new result with RRF score, preserving all fields
            fused_result = Passage(
                doc_id=result.doc_id,
                index=result.index,
                text=result.text,
                section_title=result.section_title,
                score=rrf_score,  # Use RRF-computed score
                page_url=result.page_url,
                api_name=result.api_name,
                title=result.title,
                meta=getattr(result, "meta", {}),  # Preserve metadata
                rerank_score=getattr(result, "rerank_score", None),
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