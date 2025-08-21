# src/agent/nodes/search_to_compose.py
"""
Search result processing with coverage gate integration.
Replaces the old handwritten coverage math with academic IR-style assessment.
"""

import logging
from typing import List
from src.quality.coverage import CoverageGate, Passage
from src.quality.subquery import decompose
from src.services.models import SearchResult

logger = logging.getLogger(__name__)

# Construct once (module/global scope) to reuse the model
COVERAGE = CoverageGate(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    tau=0.45,
    alpha=0.5,
    gate_ar=0.60,
    gate_andcg=0.40,
    min_actionable_spans=3,
)

# NOTE: This parallel node has been removed and integrated into the existing
# CoverageChecker in router.py to follow DRY/SOLID principles


def run_search_and_gate(user_query: str, fused_passages: list) -> dict:
    """
    fused_passages: list of dicts with keys:
      - "text" (str)
      - "url"  (str)
      - "title" (str)
      - "heading" (str)
      - "rank" (int)  # rank after your RRF
    """
    # 1) sub-queries
    subqs = decompose(user_query)

    # 2) convert to Passage objects
    passages = [
        Passage(
            text=p["text"],
            meta={k: p.get(k) for k in ("url", "title", "heading", "rank")},
        )
        for p in fused_passages
    ]

    # 3) evaluate coverage
    ev = COVERAGE.evaluate(user_query, subqs, passages)

    # 4) choose passages to send to composer: best 1 per aspect
    picks = sorted({j for idxs in ev["picks"].values() for j in idxs})
    selected = [fused_passages[j] for j in picks]

    return {
        "gate_pass": ev["gate_pass"],
        "metrics": {
            "aspect_recall": round(ev["aspect_recall"], 3),
            "alpha_ndcg": round(ev["alpha_ndcg"], 3),
            "actionable_spans": int(ev["actionable_spans"]),
            "subqueries": subqs,
            "selected_count": len(selected),
        },
        "selected_passages": selected,  # feed only these to the composer
        "all_passages_debug": fused_passages,  # keep for logs if you like
    }


def convert_search_results_to_passages(
    search_results: List[SearchResult],
) -> List[dict]:
    """
    Convert SearchResult objects to simple dicts for coverage evaluation.

    Args:
        search_results: List of SearchResult objects

    Returns:
        List of dict with keys: text, url, title, heading, rank
    """
    passages = []
    for idx, result in enumerate(search_results):
        passages.append(
            {
                "text": getattr(result, "content", "") or getattr(result, "text", ""),
                "url": getattr(result, "url", ""),
                "title": getattr(result, "title", ""),
                "heading": result.metadata.get("heading", ""),
                "rank": idx + 1,
            }
        )
    return passages
