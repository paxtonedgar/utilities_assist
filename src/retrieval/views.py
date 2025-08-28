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
from src.services.models import Passage
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
    return await _run_search_view(
        query=query,
        view_type="info",
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=top_k,
    )


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
    return await _run_search_view(
        query=query,
        view_type="procedure",
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=top_k,
    )


async def _run_search_view(
    query: str,
    view_type: str,
    search_client=None,
    embed_client=None,
    embed_model: str = "text-embedding-ada-002",
    top_k: int = 10,
) -> Optional[ViewResult]:
    """Common search view implementation for both info and procedure views."""
    start_time = time.time()
    
    try:
        # Setup search clients and resources
        search_client, embed_client, embed_model, index_name = _setup_search_resources(
            search_client, embed_client, embed_model
        )
        
        # Get filters based on query and view type
        filters, is_utility_query = _get_search_filters(query, view_type)
        
        logger.info(f"Running {view_type} view for query: '{query[:50]}...'")
        
        # Execute search with fallback handling
        result, fused_results = await _execute_search_with_fallback(
            query, filters, is_utility_query, view_type,
            search_client, embed_client, embed_model, index_name
        )
        
        # Process results and build response
        return _build_view_result(
            result, fused_results, view_type, index_name, start_time, top_k
        )
        
    except Exception as e:
        return _handle_view_error(e, view_type, start_time)


def _setup_search_resources(search_client, embed_client, embed_model):
    """Setup search resources with fallback to default resources."""
    if not search_client or not embed_client:
        resources = get_resources()
        search_client = search_client or resources.search_client
        embed_client = embed_client or resources.embed_client
        embed_model = embed_model or resources.settings.embed.model
    
    index_name = OpenSearchConfig.get_default_index()
    return search_client, embed_client, embed_model, index_name


def _get_search_filters(query: str, view_type: str):
    """Get search filters based on query type and view type."""
    req_id = get_or_create_req_id()
    
    # Check if this is a utility-related query
    query_lower = query.lower()
    is_utility_query = _is_utility_query(query_lower)
    
    # Determine intent type and get filters
    intent_type = "utilities" if is_utility_query else "confluence"
    filter_state = get_consistent_filters(req_id, intent_type=intent_type, view_type=view_type)
    filters = filter_state.to_opensearch_filters()
    
    if is_utility_query:
        logger.info(f"Detected utility query in {view_type} view - using broader filters")
    
    return filters, is_utility_query


def _is_utility_query(query_lower: str) -> bool:
    """Check if query is utility-related."""
    utility_terms = [
        "ciu", "customer interaction utility", "etu", 
        "enhanced transaction", "utility", "utilities"
    ]
    return any(term in query_lower for term in utility_terms)


async def _execute_search_with_fallback(
    query: str, filters, is_utility_query: bool, view_type: str,
    search_client, embed_client, embed_model: str, index_name: str
):
    """Execute search with fallback for utility queries."""
    # Initial search
    result = await search_index_tool(
        index=index_name,
        query=query,
        filters=filters,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=36,
        strategy="enhanced_rrf",
    )
    
    fused_results = _dedupe_by_doc_id(result.results)
    
    # Fallback for utility queries with insufficient results
    if is_utility_query and len(fused_results) < 5:
        result, fused_results = await _execute_fallback_search(
            query, view_type, search_client, embed_client, embed_model, 
            index_name, result, fused_results
        )
    
    return result, fused_results


async def _execute_fallback_search(
    query: str, view_type: str, search_client, embed_client, 
    embed_model: str, index_name: str, original_result, original_fused_results
):
    """Execute fallback search without filters for better coverage."""
    logger.warning(
        f"Utility query in {view_type} view returned only {len(original_fused_results)} results. "
        "Retrying without filters for broader coverage."
    )
    
    fallback_result = await search_index_tool(
        index=index_name,
        query=query,
        filters=None,
        search_client=search_client,
        embed_client=embed_client,
        embed_model=embed_model,
        top_k=50,
        strategy="enhanced_rrf",
    )
    
    # Merge results
    merged_results = _merge_search_results(fallback_result.results, original_fused_results)
    fallback_result.results = merged_results
    fused_results = _dedupe_by_doc_id(merged_results)
    
    logger.info(
        f"Fallback search in {view_type} view yielded {len(fused_results)} unique results"
    )
    
    return fallback_result, fused_results


def _merge_search_results(fallback_results: List[Passage], original_results: List[Passage]) -> List[Passage]:
    """Merge fallback results with original results, avoiding duplicates."""
    seen_doc_ids = {r.doc_id for r in fallback_results}
    merged_results = list(fallback_results)
    
    for orig_result in original_results:
        if orig_result.doc_id not in seen_doc_ids:
            merged_results.append(orig_result)
    
    return merged_results


def _build_view_result(
    result, fused_results: List[Passage], view_type: str, 
    index_name: str, start_time: float, top_k: int
) -> ViewResult:
    """Build ViewResult from search results."""
    # Determine reranked results
    reranked_results = None
    if hasattr(result, "method") and "ce" in result.method:
        reranked_results = fused_results
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Build metrics
    metrics = _build_view_metrics(result, reranked_results, view_type, index_name, elapsed_ms)
    
    # Log completion
    log_event(stage=f"{view_type}_view", event="success", ms=elapsed_ms, **metrics)
    
    return ViewResult(
        view=view_type,
        fused_top8=fused_results[:8],
        reranked_topk=reranked_results[:top_k] if reranked_results else None,
        metrics=metrics,
    )


def _build_view_metrics(
    result, reranked_results: Optional[List[Passage]], 
    view_type: str, index_name: str, elapsed_ms: float
) -> dict:
    """Build metrics dictionary for view result."""
    return {
        "view": view_type,
        "index": index_name,
        "bm25_hits": result.diagnostics.get("bm25_hits", 0),
        "knn_hits": result.diagnostics.get("knn_hits", 0),
        "rrf_candidates": result.diagnostics.get("rrf_candidates", 0),
        "ce_kept": len(reranked_results) if reranked_results else 0,
        "timeline_ms": elapsed_ms,
        "method": result.method,
        "unique_docs_pre_rerank": result.diagnostics.get("unique_docs_pre_rerank", 0),
        "cross_encoder_skipped": result.diagnostics.get("cross_encoder_skipped", False),
    }


def _handle_view_error(e: Exception, view_type: str, start_time: float) -> None:
    """Handle view execution error."""
    elapsed_ms = (time.time() - start_time) * 1000
    logger.error(f"{view_type.capitalize()} view failed after {elapsed_ms:.0f}ms: {e}")
    
    log_event(
        stage=f"{view_type}_view",
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
