# src/retrieval/views.py
"""
Search view builders for info and procedure queries.
Wraps existing search infrastructure with structured result formatting.
"""

import logging
from typing import List, Optional
import time

from .actionability import ViewResult
from src.agent.tools.search import search_index_tool
from src.services.models import SearchResult as Passage
from src.infra.resource_manager import get_resources
from src.infra.search_config import OpenSearchConfig
from src.telemetry.logger import log_event, get_or_create_req_id
from src.util.filters import get_consistent_filters

logger = logging.getLogger(__name__)


async def run_info_view(
    query: str,
    search_client=None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 10,
) -> Optional[ViewResult]:
    """
    Run info-focused search view for definition/explanation queries.

    Args:
        query: User query text
        search_client: OpenSearch client
        embed_client: Embedding client
        embed_model: Embedding model name
        top_k: Number of results to return

    Returns:
        ViewResult with info-focused search results
    """
    start_time = time.time()

    try:
        # Get resources if not provided
        if not search_client or not embed_client:
            resources = get_resources()
            search_client = search_client or resources.search_client
            embed_client = embed_client or resources.embed_client
            embed_model = embed_model or resources.settings.embed.model

        # Use default index for broad info coverage
        index_name = OpenSearchConfig.get_default_index()

        # Get consistent filters (won't flip during rewrite loops)
        req_id = get_or_create_req_id()
        filter_state = get_consistent_filters(
            req_id, intent_type="confluence", view_type="info"
        )
        filters = filter_state.to_opensearch_filters()

        logger.info(f"Running info view for query: '{query[:50]}...'")

        # Run enhanced RRF search via existing tool
        result = await search_index_tool(
            index=index_name,
            query=query,
            filters=filters,
            search_client=search_client,
            embed_client=embed_client,
            embed_model=embed_model,
            top_k=36,  # Get more candidates for better info coverage
            strategy="enhanced_rrf",
        )

        # Extract results with deduplication by doc_id
        fused_results = _dedupe_by_doc_id(result.results)

        # Determine reranked results (may be None if CE timed out)
        reranked_results = None
        if hasattr(result, "method") and "ce" in result.method:
            reranked_results = fused_results  # Already reranked

        elapsed_ms = (time.time() - start_time) * 1000

        # Build metrics
        metrics = {
            "view": "info",
            "index": index_name,
            "bm25_hits": result.diagnostics.get("bm25_hits", 0),
            "knn_hits": result.diagnostics.get("knn_hits", 0),
            "rrf_candidates": result.diagnostics.get("rrf_candidates", 0),
            "ce_kept": len(reranked_results) if reranked_results else 0,
            "timeline_ms": elapsed_ms,
            "method": result.method,
            "unique_docs_pre_rerank": result.diagnostics.get(
                "unique_docs_pre_rerank", 0
            ),
            "cross_encoder_skipped": result.diagnostics.get(
                "cross_encoder_skipped", False
            ),
        }

        # Log view completion
        log_event(stage="info_view", event="success", ms=elapsed_ms, **metrics)

        return ViewResult(
            view="info",
            fused_top8=fused_results[:8],
            reranked_topk=reranked_results[:top_k] if reranked_results else None,
            metrics=metrics,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Info view failed after {elapsed_ms:.0f}ms: {e}")

        log_event(
            stage="info_view",
            event="error",
            ms=elapsed_ms,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
        )

        return None


async def run_procedure_view(
    query: str,
    search_client=None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 10,
) -> Optional[ViewResult]:
    """
    Run procedure-focused search view for how-to/onboarding queries.

    Args:
        query: User query text
        search_client: OpenSearch client
        embed_client: Embedding client
        embed_model: Embedding model name
        top_k: Number of results to return

    Returns:
        ViewResult with procedure-focused search results
    """
    start_time = time.time()

    try:
        # Get resources if not provided
        if not search_client or not embed_client:
            resources = get_resources()
            search_client = search_client or resources.search_client
            embed_client = embed_client or resources.embed_client
            embed_model = embed_model or resources.settings.embed.model

        # Use default index but with procedure-focused filters
        index_name = OpenSearchConfig.get_default_index()

        # Get consistent filters (won't flip during rewrite loops)
        req_id = get_or_create_req_id()
        filter_state = get_consistent_filters(
            req_id, intent_type="confluence", view_type="procedure"
        )
        filters = filter_state.to_opensearch_filters()

        logger.info(f"Running procedure view for query: '{query[:50]}...'")

        # Run enhanced RRF search via existing tool
        result = await search_index_tool(
            index=index_name,
            query=query,
            filters=filters,
            search_client=search_client,
            embed_client=embed_client,
            embed_model=embed_model,
            top_k=36,  # Get more candidates for better procedure detection
            strategy="enhanced_rrf",
        )

        # Extract results with deduplication by doc_id
        fused_results = _dedupe_by_doc_id(result.results)

        # Determine reranked results (may be None if CE timed out)
        reranked_results = None
        if hasattr(result, "method") and "ce" in result.method:
            reranked_results = fused_results  # Already reranked

        elapsed_ms = (time.time() - start_time) * 1000

        # Build metrics
        metrics = {
            "view": "procedure",
            "index": index_name,
            "bm25_hits": result.diagnostics.get("bm25_hits", 0),
            "knn_hits": result.diagnostics.get("knn_hits", 0),
            "rrf_candidates": result.diagnostics.get("rrf_candidates", 0),
            "ce_kept": len(reranked_results) if reranked_results else 0,
            "timeline_ms": elapsed_ms,
            "method": result.method,
            "unique_docs_pre_rerank": result.diagnostics.get(
                "unique_docs_pre_rerank", 0
            ),
            "cross_encoder_skipped": result.diagnostics.get(
                "cross_encoder_skipped", False
            ),
        }

        # Log view completion
        log_event(stage="procedure_view", event="success", ms=elapsed_ms, **metrics)

        return ViewResult(
            view="procedure",
            fused_top8=fused_results[:8],
            reranked_topk=reranked_results[:top_k] if reranked_results else None,
            metrics=metrics,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Procedure view failed after {elapsed_ms:.0f}ms: {e}")

        log_event(
            stage="procedure_view",
            event="error",
            ms=elapsed_ms,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
        )

        return None


def _dedupe_by_doc_id(results: List[Passage]) -> List[Passage]:
    """
    Deduplicate search results by document ID, keeping highest scoring.

    Args:
        results: List of search results

    Returns:
        Deduplicated list of results
    """
    if not results:
        return results

    seen_doc_ids = set()
    deduped = []

    # Results should already be score-sorted, so first occurrence = highest score
    for result in results:
        if result.doc_id not in seen_doc_ids:
            deduped.append(result)
            seen_doc_ids.add(result.doc_id)

    return deduped
