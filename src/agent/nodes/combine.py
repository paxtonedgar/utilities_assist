"""Streamlined combine node - simplified merging without over-engineering."""

import logging
from typing import List
from src.services.models import Passage
from .base_node import to_state_dict, from_state_dict

logger = logging.getLogger(__name__)


async def combine_node(state, config=None, *, store=None):
    """
    Combine and merge results from searches with minimal complexity.

    Simplified from 477 lines to ~80 lines by removing:
    - Complex domain boosting logic
    - Elaborate quality gates
    - Multi-search fusion complexity
    - MMR diversification
    """
    incoming_type = type(state)

    try:
        # Convert to dict for processing
        s = to_state_dict(state)

        logger.info(
            "NODE_START combine | keys=%s | search_results_count=%d",
            list(s.keys()),
            len(s.get("search_results", [])),
        )

        all_results = s.get("search_results", [])
        query = s.get("normalized_query", s.get("original_query", ""))

        if not all_results:
            logger.warning("No search results to combine")
            return _create_no_results_response(s, incoming_type, query)

        # Simple deduplication and sorting - no complex fusion
        combined_results = _deduplicate_and_sort(all_results)

        # Limit to reasonable number of results
        if len(combined_results) > 12:
            combined_results = combined_results[:12]

        # Simple quality check
        if len(combined_results) < 2:
            logger.info(
                f"Low document count ({len(combined_results)}) for query: {query[:50]}..."
            )

        # Debug logging
        if combined_results:
            top_results_summary = [
                (r.meta.get("title", f"doc_{r.doc_id[:8]}"), round(float(r.score), 3))
                for r in combined_results[:5]
            ]
            logger.info(f"COMBINED_TOP: {top_results_summary}")

        # Build context
        final_context = _build_simple_context(combined_results)

        logger.info(
            f"NODE_END combine | combined_results={len(combined_results)} | context_length={len(final_context)}"
        )

        # Return state update
        merged = {
            **s,
            "combined_results": combined_results,
            "final_context": final_context,
            "workflow_path": s.get("workflow_path", []) + ["combine"],
        }
        return from_state_dict(incoming_type, merged)

    except Exception as e:
        logger.error(f"Combine node failed: {e}")
        return from_state_dict(
            incoming_type,
            {**s, "error_messages": s.get("error_messages", []) + [str(e)]},
        )


def _create_no_results_response(s: dict, incoming_type, query: str):
    """Create response when no results found."""
    # Generate search suggestions
    suggestions = [
        f"{query} setup documentation",
        f"{query} API reference",
        f"{query} integration guide",
    ]

    suggestion_text = "Try these searches: " + " | ".join(f'"{s}"' for s in suggestions)

    merged = {
        **s,
        "combined_results": [],
        "final_context": f"No documentation found for '{query}'.",
        "final_answer": f"No relevant documentation found. {suggestion_text}",
        "workflow_path": s.get("workflow_path", []) + ["combine", "no_results"],
    }
    return from_state_dict(incoming_type, merged)


def _deduplicate_and_sort(results: List[Passage]) -> List[Passage]:
    """Simple deduplication by doc_id and sort by score."""
    seen_ids = set()
    deduped = []

    for result in results:
        if result.doc_id not in seen_ids:
            seen_ids.add(result.doc_id)
            deduped.append(result)

    # Sort by relevance score (already set by reranker)
    return sorted(deduped, key=lambda x: x.score, reverse=True)


def _build_simple_context(results: List[Passage], max_length: int = 8000) -> str:
    """Build simple context without complex formatting."""
    if not results:
        return ""

    context_parts = []
    current_length = 0

    for i, result in enumerate(results, 1):
        # Simple format: [Source N: Title] Content
        title = result.meta.get("title", f"Document {i}")
        doc_context = f"[Source {i}: {title}]\n{result.text.strip()}\n"

        if current_length + len(doc_context) > max_length:
            break

        context_parts.append(doc_context)
        current_length += len(doc_context)

    context = "\n".join(context_parts)

    if current_length >= max_length:
        context += (
            f"\n[Note: Context truncated - showing top {len(context_parts)} results]"
        )

    return context
