"""
Document-by-document corpus scan using OpenSearch PIT.

Iterates the entire index (or up to --limit docs), extracts steps per section,
builds intra-doc NEXT edges, and domain relations (teams/divisions/apps/tools/platforms/diagrams).
Writes NDJSON outputs under --out-dir for later loading/visualization.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from src.infra.opensearch_client import OpenSearchClient
from src.ontology.extractors import regex_step_extractor
from src.ontology.domain_extractors import extract_domain_entities_and_relations
from src.ontology.signals import compute_signals_for_pair
from src.ontology.scoring import score_edge


def _passages_from_hit(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split a hit into section-level passages to maximize nugget discovery."""
    src = hit.get("_source", {})
    passages: List[Dict[str, Any]] = []
    doc_id = hit.get("_id")
    idx = hit.get("_index")

    # Prefer nested sections if present
    sections = src.get("sections")
    if isinstance(sections, list) and sections:
        for s in sections:
            if isinstance(s, dict):
                text = s.get("content") or s.get("text") or s.get("body")
                if text:
                    passages.append(
                        {
                            "doc_id": doc_id,
                            "index": idx,
                            "section_title": s.get("title") or s.get("heading"),
                            "text": str(text),
                            "page_url": src.get("page_url"),
                            "title": src.get("title"),
                        }
                    )
    else:
        # Fallback to document-level fields
        for f in ("body", "content", "text", "description", "summary"):
            if src.get(f):
                passages.append(
                    {
                        "doc_id": doc_id,
                        "index": idx,
                        "section_title": None,
                        "text": str(src[f]),
                        "page_url": src.get("page_url"),
                        "title": src.get("title"),
                    }
                )
                break

    return passages


def _build_next_edges(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    ordered = sorted([s for s in steps if isinstance(s.get("order"), int)], key=lambda x: x["order"])
    for a, b in zip(ordered, ordered[1:]):
        pair = {
            "type": "NEXT",
            "a": a,
            "b": b,
            "doc_id_a": a.get("doc_id"),
            "doc_id_b": b.get("doc_id"),
            "order_a": a.get("order"),
            "order_b": b.get("order"),
            "evidence_refs": [
                {"doc_id": a.get("doc_id"), "snippet": a.get("evidence", {}).get("text_snippet")},
                {"doc_id": b.get("doc_id"), "snippet": b.get("evidence", {}).get("text_snippet")},
            ],
        }
        edges.append(pair)
    return edges


def process_doc(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Extract steps and relations for a single document hit.

    Returns minimal dict with steps, edges, and meta for the doc.
    """
    passages = _passages_from_hit(hit)
    doc_id = hit.get("_id")
    out_steps: List[Dict[str, Any]] = []
    domain_edges: List[Dict[str, Any]] = []

    # Per-section extraction to capture nuggets
    for p in passages:
        steps = regex_step_extractor(p.get("text", ""))
        for s in steps:
            s.update({
                "doc_id": p.get("doc_id"),
                "index": p.get("index"),
                "section_title": p.get("section_title"),
                "page_url": p.get("page_url"),
            })
        out_steps.extend(steps)

        _, rels = extract_domain_entities_and_relations(p)
        domain_edges.extend(rels)

    # Build NEXT edges within doc
    next_edges = _build_next_edges(out_steps)

    # Score edges (NEXT + domain)
    scored: List[Dict[str, Any]] = []
    all_edges = next_edges + domain_edges
    for e in all_edges:
        rel = e.get("type", "NEXT")
        if rel == "NEXT":
            sigs = compute_signals_for_pair(e["a"], e["b"])  # lightweight signals
            e_score = score_edge(relation=rel, pair={**e, "signals": sigs})
        else:
            e_score = score_edge(relation=rel, pair=e)
        scored.append({
            "type": rel,
            "score": e_score.score,
            "accepted": e_score.accepted,
            "signals": e_score.signals,
            "notes": e_score.notes,
        })

    return {
        "doc_id": doc_id,
        "index": hit.get("_index"),
        "steps": out_steps,
        "edges": all_edges,
        "scored": scored,
    }


def run(index: str | None, limit: int, batch: int, out_dir: str):
    client = OpenSearchClient()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    steps_f = (out_path / "steps.ndjson").open("w", encoding="utf-8")
    edges_f = (out_path / "edges.ndjson").open("w", encoding="utf-8")
    meta_f = (out_path / "docs.ndjson").open("w", encoding="utf-8")

    try:
        count = 0
        for hit in client.iterate_index(index=index, fields=None, batch_size=batch, max_docs=limit):
            result = process_doc(hit)
            # Write per-doc lines
            meta_f.write(json.dumps({"doc_id": result["doc_id"], "index": result["index"], "steps": len(result["steps"]), "edges": len(result["edges"])}) + "\n")
            for s in result["steps"]:
                steps_f.write(json.dumps({"doc_id": result["doc_id"], **s}) + "\n")
            for e, s in zip(result["edges"], result["scored"]):
                edges_f.write(json.dumps({"doc_id": result["doc_id"], **e, "score": s.get("score"), "accepted": s.get("accepted"), "signals": s.get("signals")}) + "\n")

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} docs...")
        print(f"Done. Processed {count} documents. Outputs in {out_path}")
    finally:
        steps_f.close(); edges_f.close(); meta_f.close()


def main():
    ap = argparse.ArgumentParser(description="Document-by-document ontology scan")
    ap.add_argument("--index", type=str, default=None, help="Index or alias (defaults to config alias)")
    ap.add_argument("--limit", type=int, default=1000, help="Max docs to process (0 = all)")
    ap.add_argument("--batch", type=int, default=200, help="Batch size for PIT pagination")
    ap.add_argument("--out-dir", type=str, default="outputs/ontology_scan", help="Output directory")
    args = ap.parse_args()

    run(index=args.index, limit=(None if args.limit == 0 else args.limit), batch=args.batch, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

