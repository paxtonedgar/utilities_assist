"""
Cross-encoder reranking policy - A3 logic for smart CE usage.

This module implements intelligent policies for when to use expensive cross-encoder
reranking vs when to skip it and preserve RRF results.
"""

import logging
import signal
import time
from contextlib import contextmanager
from typing import List
from src.services.models import RankedHit, RerankResult
from src.telemetry.logger import log_event

logger = logging.getLogger(__name__)


class RerankerTimeoutError(Exception):
    """Raised when cross-encoder reranking times out."""

    pass


@contextmanager
def timeout_guard(seconds: float):
    """Context manager for timing out long-running operations."""

    def timeout_handler(signum, frame):
        raise RerankerTimeoutError(f"Operation timed out after {seconds}s")

    # Set the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def maybe_rerank(
    rrf_hits: List[RankedHit],
    query: str,
    ce_model,
    max_candidates: int = 12,
    timeout_ms: int = 8000,
    min_return: int = 3,
    skip_definitional: bool = True,
) -> RerankResult:
    """
    A3: Smart cross-encoder reranking with policy decisions.

    Decides whether to use expensive CE reranking or preserve RRF results
    based on query characteristics and risk factors.

    Args:
        rrf_hits: RRF-ranked hits with passages
        query: User query string
        ce_model: Cross-encoder model for reranking
        max_candidates: Maximum hits to send to CE
        timeout_ms: Timeout in milliseconds
        min_return: Minimum results to return (avoid collapse)
        skip_definitional: Skip CE for definitional queries

    Returns:
        RerankResult with reranked items and decision reasoning
    """
    start_time = time.time()

    # A3: Skip definitional queries (cheap heuristic)
    if skip_definitional and _is_definitional(query):
        log_event(
            stage="rerank_skip",
            reason="definitional",
            query_length=len(query.split()),
            query=query[:50],
        )
        return RerankResult(rrf_hits, False, "skipped_definitional")

    # Prepare candidates for CE
    candidates = rrf_hits[:max_candidates]

    if len(candidates) < 2:
        log_event(
            stage="rerank_skip",
            reason="insufficient_candidates",
            candidate_count=len(candidates),
        )
        return RerankResult(rrf_hits, False, "insufficient_candidates")

    try:
        # A3: Apply timeout guard
        with timeout_guard(timeout_ms / 1000.0):
            log_event(
                stage="rerank_start",
                candidate_count=len(candidates),
                timeout_ms=timeout_ms,
            )

            # Extract texts for CE scoring
            texts = []
            for rh in candidates:
                if rh.passages:
                    # Use first passage text
                    texts.append(rh.passages[0].text)
                else:
                    # Fallback to hit source if no passages
                    source = rh.hit.get("_source", {})
                    text = (
                        source.get("title", "")
                        + " "
                        + source.get("api_name", "")
                        + " "
                        + source.get("utility_name", "")
                    ).strip()
                    texts.append(text or "No content available")

            # Call cross-encoder (adapt to your existing CE interface)
            ce_scores = ce_model.predict([(query, text) for text in texts])

            # Create score tuples and sort by CE score
            scored_hits = list(zip(candidates, ce_scores))
            scored_hits.sort(key=lambda x: x[1], reverse=True)

            # Extract reranked hits
            reranked_hits = [hit for hit, score in scored_hits]

        elapsed_ms = (time.time() - start_time) * 1000

        # A3: Check for collapse (too few results after reranking)
        if len(reranked_hits) < min_return:
            log_event(
                stage="rerank_collapse",
                original_count=len(rrf_hits),
                reranked_count=len(reranked_hits),
                min_return=min_return,
                elapsed_ms=elapsed_ms,
            )
            return RerankResult(rrf_hits, False, "collapse")

        log_event(
            stage="rerank_success",
            candidate_count=len(candidates),
            result_count=len(reranked_hits),
            elapsed_ms=elapsed_ms,
        )

        return RerankResult(reranked_hits, True, "ok")

    except RerankerTimeoutError:
        elapsed_ms = (time.time() - start_time) * 1000
        log_event(
            stage="rerank_timeout",
            timeout_ms=timeout_ms,
            elapsed_ms=elapsed_ms,
            candidate_count=len(candidates),
        )
        return RerankResult(rrf_hits, False, "timeout")

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Cross-encoder reranking failed: {e}")
        log_event(
            stage="rerank_error",
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            elapsed_ms=elapsed_ms,
        )
        return RerankResult(rrf_hits, False, "error")


def _is_definitional(query: str) -> bool:
    """
    Detect definitional queries that don't benefit from CE reranking.

    These are typically short, simple queries asking "what is X?" that
    have clear factual answers where keyword matching (BM25/RRF) is sufficient.

    Args:
        query: User query string

    Returns:
        True if query appears to be definitional
    """
    # Normalize query
    tokens = query.lower().strip().split()

    # Very short queries (≤3 tokens) are often definitional
    if len(tokens) <= 3:
        return True

    # Pattern-based detection
    definitional_patterns = [
        # What is X?
        lambda t: len(t) >= 2 and t[0] == "what" and t[1] in ["is", "are"],
        # Define X, Definition of X
        lambda t: t[0] in ["define", "definition"],
        # X meaning, X stands for
        lambda t: "meaning" in t or "stands" in t,
        # Single word/acronym queries
        lambda t: len(t) == 1 and len(t[0]) <= 5,
    ]

    for pattern in definitional_patterns:
        if pattern(tokens):
            return True

    return False






