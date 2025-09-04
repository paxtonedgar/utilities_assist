"""
Local pipeline to fetch passages from OpenSearch, extract steps, build intra-doc
edge candidates, score, and run QC — without persisting a graph.

Usage (CLI):
    python -m src.ontology.pipeline --query "install OR configure" --max-docs 10 --csv-out qc_edges.csv
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import argparse
import logging

from src.infra.opensearch_client import OpenSearchClient
from src.ontology.extractors import regex_step_extractor
from src.ontology.domain_extractors import extract_domain_entities_and_relations
from src.ontology.signals import compute_signals_for_pair
from src.ontology.scoring import score_edge
from src.ontology.qc import qc_report


logger = logging.getLogger(__name__)


def fetch_passages(query: str, k: int = 25) -> List[Dict[str, Any]]:
    client = OpenSearchClient()
    res = client.bm25_search(query=query, k=k)
    # Convert Passage objects to serializable dicts we need
    passages: List[Dict[str, Any]] = []
    for p in res.results:
        passages.append(
            {
                "doc_id": p.doc_id,
                "index": p.index,
                "section_title": p.section_title,
                "text": p.text,
                "page_url": p.page_url,
                "title": p.title,
                "score": p.score,
            }
        )
    return passages


def extract_steps_from_passages(passages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract steps per doc_id; returns mapping doc_id -> list[step dict]."""
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for p in passages:
        steps = regex_step_extractor(p.get("text", ""))
        # add provenance
        for s in steps:
            s["doc_id"] = p.get("doc_id")
            s["index"] = p.get("index")
            s["section_title"] = p.get("section_title")
            s["page_url"] = p.get("page_url")
        if steps:
            by_doc.setdefault(p.get("doc_id"), []).extend(steps)
    return by_doc


def build_next_candidates(steps_by_doc: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Build intra-doc NEXT edge candidates based on order fields when present."""
    edges: List[Dict[str, Any]] = []
    for doc_id, steps in steps_by_doc.items():
        steps_sorted = sorted(
            [s for s in steps if isinstance(s.get("order"), int)], key=lambda x: x["order"]
        )
        for a, b in zip(steps_sorted, steps_sorted[1:]):
            pair = {
                "a": a,
                "b": b,
                "doc_id_a": a.get("doc_id"),
                "doc_id_b": b.get("doc_id"),
                "type": "NEXT",
                "order_a": a.get("order"),
                "order_b": b.get("order"),
                "evidence_refs": [
                    {"doc_id": a.get("doc_id"), "snippet": a.get("evidence", {}).get("text_snippet")},
                    {"doc_id": b.get("doc_id"), "snippet": b.get("evidence", {}).get("text_snippet")},
                ],
            }
            edges.append(pair)
    return edges


def score_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for e in edges:
        sigs = compute_signals_for_pair(e["a"], e["b"])
        e_score = score_edge(relation=e.get("type", "NEXT"), pair={**e, "signals": sigs})
        scored.append(
            {
                "relation": e_score.relation,
                "score": e_score.score,
                "threshold": e_score.threshold,
                "accepted": e_score.accepted,
                "signals": e_score.signals,
                "notes": e_score.notes,
            }
        )
    return scored


def run(query: str, max_docs: int = 25, csv_out: Optional[str] = None) -> None:
    passages = fetch_passages(query, k=max_docs)
    steps_by_doc = extract_steps_from_passages(passages)
    next_edges = build_next_candidates(steps_by_doc)
    scored_next = score_edges(next_edges)

    # Domain extraction (teams/divisions/apps/tools/platforms/diagrams)
    domain_edges: List[Dict[str, Any]] = []
    for p in passages:
        _, rels = extract_domain_entities_and_relations(p)
        domain_edges.extend(rels)
    # Score domain edges
    scored_domain: List[Dict[str, Any]] = []
    for e in domain_edges:
        rel = e.get("type", "OWNS")
        e_score = score_edge(relation=rel, pair=e)
        scored_domain.append(
            {
                "relation": e_score.relation,
                "score": e_score.score,
                "threshold": e_score.threshold,
                "accepted": e_score.accepted,
                "signals": e_score.signals,
                "notes": e_score.notes,
            }
        )

    # Combine for QC reporting (edges and scores must be zipped by index)
    all_edges = next_edges + domain_edges
    all_scored = scored_next + scored_domain
    result = qc_report(all_edges, all_scored, csv_out=csv_out)

    logger.info(
        "QC: total=%d accepted=%d rejected=%d violations=%d",
        result.total_edges,
        result.accepted_edges,
        result.rejected_edges,
        len(result.violations),
    )
    if result.violations:
        logger.info("Violations: %s", ", ".join(result.violations[:20]))


def main():
    parser = argparse.ArgumentParser(description="Ontology pipeline (local probe)")
    parser.add_argument("--query", type=str, default="install OR configure", help="BM25 query")
    parser.add_argument("--max-docs", type=int, default=25, help="Max passages to fetch")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV path for scored edges")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(query=args.query, max_docs=args.max_docs, csv_out=args.csv_out)


if __name__ == "__main__":
    main()
