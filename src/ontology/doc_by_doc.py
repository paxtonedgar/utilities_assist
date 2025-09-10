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
from typing import Dict, Any, List, Tuple

from src.infra.opensearch_client import OpenSearchClient
from src.ontology.extractors import regex_step_extractor
from src.ontology.gazetteer import Gazetteer
import hashlib


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
                # Swagger-style: method + endpoint present in section
                method = (s.get("method") or s.get("http_method") or "").strip()
                endpoint = (s.get("endpoint") or s.get("path") or "").strip()
                if method and endpoint:
                    line = f"{method.upper()} {endpoint}"
                    passages.append(
                        {
                            "doc_id": doc_id,
                            "index": idx,
                            "section_title": s.get("summary") or s.get("title") or s.get("heading"),
                            "text": line,
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

        # Swagger-style: top-level method + endpoint
        method = (src.get("method") or src.get("http_method") or "").strip()
        endpoint = (src.get("endpoint") or src.get("path") or "").strip()
        if method and endpoint:
            line = f"{method.upper()} {endpoint}"
            passages.append(
                {
                    "doc_id": doc_id,
                    "index": idx,
                    "section_title": src.get("summary") or src.get("title"),
                    "text": line,
                    "page_url": src.get("page_url"),
                    "title": src.get("title"),
                }
            )

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
            "evidence_refs": [],  # populated later
            "accepted": False,
        }
        edges.append(pair)
    return edges


def _find_offsets(haystack: str, needle: str) -> Dict[str, int] | None:
    if not haystack or not needle:
        return None
    start = haystack.find(needle)
    if start < 0:
        start = haystack.lower().find(needle.lower())
    if start < 0:
        return None
    return {"start": start, "end": start + len(needle)}


def process_doc(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Extract steps and relations for a single document hit.

    Returns minimal dict with steps, edges, and meta for the doc.
    """
    passages = _passages_from_hit(hit)
    doc_id = hit.get("_id")
    out_steps: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    gaz = Gazetteer()

    # Per-section extraction to capture nuggets
    for p in passages:
        section_text = p.get("text", "")
        section_hash = _sha256(section_text)
        steps = regex_step_extractor(section_text)
        for s in steps:
            s.update({
                "doc_id": p.get("doc_id"),
                "index": p.get("index"),
                "section_title": p.get("section_title"),
                "page_url": p.get("page_url"),
                "section_hash": section_hash,
            })
            snippet = (s.get("evidence") or {}).get("text_snippet", "")
            offs = _find_offsets(section_text, snippet)
            if offs:
                s["evidence"]["offsets"] = offs
        out_steps.extend(steps)
        # Gazetteer mentions (no relations)
        for etype, name in gaz.find_mentions(section_text):
            entities.append({"type": etype, "name": name})

    # Build NEXT edges within doc and attach evidence with hashes/offsets
    next_edges = _build_next_edges(out_steps)
    for e in next_edges:
        a, b = e.get("a", {}), e.get("b", {})
        ev = []
        for step in (a, b):
            ev.append({
                "doc_id": step.get("doc_id"),
                "snippet": (step.get("evidence") or {}).get("text_snippet"),
                "offsets": (step.get("evidence") or {}).get("offsets"),
                "content_hash": step.get("section_hash"),
            })
        e["evidence_refs"] = ev

    return {
        "doc_id": doc_id,
        "index": hit.get("_index"),
        "steps": out_steps,
        "edges": next_edges,
        "entities": sorted({(x["type"], x["name"]) for x in entities}),
    }


def _load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _append_processed_id(path: Path, doc_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(doc_id + "\n")


def _update_summary(summary: Dict[str, Any], doc_result: Dict[str, Any]) -> None:
    summary["docs"] += 1
    summary["steps"] += len(doc_result.get("steps", []))
    summary["edges_total"] += len(doc_result.get("edges", []))
    # Relation counts
    for e, s in zip(doc_result.get("edges", []), doc_result.get("scored", [])):
        rel = e.get("type", "?")
        bucket = summary["relations"].setdefault(rel, {"total": 0, "accepted": 0})
        bucket["total"] += 1
        if s.get("accepted"):
            bucket["accepted"] += 1
    # Top docs by steps
    summary.setdefault("top_docs", []).append({
        "doc_id": doc_result.get("doc_id"),
        "index": doc_result.get("index"),
        "steps": len(doc_result.get("steps", [])),
        "edges": len(doc_result.get("edges", [])),
    })


def _finalize_summary(summary: Dict[str, Any], out_path: Path) -> None:
    # Keep top 25 docs by steps
    summary["top_docs"] = sorted(summary["top_docs"], key=lambda r: r.get("steps", 0), reverse=True)[:25]
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run(index: str | None, limit: int, batch: int, out_dir: str, resume: bool, checkpoint_file: str, write_summary: bool):
    client = OpenSearchClient()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    steps_f = (out_path / "steps.ndjson").open("w", encoding="utf-8")
    edges_f = (out_path / "edges.ndjson").open("w", encoding="utf-8")
    meta_f = (out_path / "docs.ndjson").open("w", encoding="utf-8")
    ents_f = (out_path / "entities.ndjson").open("w", encoding="utf-8")
    ckpt_path = Path(checkpoint_file) if checkpoint_file else (out_path / "checkpoint.ids")
    # Load previously processed ids if resuming; otherwise start empty
    processed_ids = _load_processed_ids(ckpt_path) if resume else set()
    # Track seen ids for this run as well to deduplicate non-PIT iteration
    seen_ids = set(processed_ids)

    summary: Dict[str, Any] = {"docs": 0, "steps": 0, "edges_total": 0, "relations": {}, "top_docs": []}

    try:
        count = 0
        dupes = 0
        for hit in client.iterate_index(index=index, fields=None, batch_size=batch, max_docs=limit):
            hid = hit.get("_id")
            if hid in seen_ids:
                dupes += 1
                continue
            result = process_doc(hit)
            # Write per-doc lines
            meta_f.write(json.dumps({"doc_id": result["doc_id"], "index": result["index"], "steps": len(result["steps"]), "edges": len(result["edges"])}) + "\n")
            for s in result["steps"]:
                steps_f.write(json.dumps({"doc_id": result["doc_id"], **s}) + "\n")
            for e in result["edges"]:
                e_out = {"doc_id": result["doc_id"], **e, "accepted": False}
                edges_f.write(json.dumps(e_out) + "\n")
            for t, n in result.get("entities", []):
                ents_f.write(json.dumps({"doc_id": result["doc_id"], "type": t, "name": n}) + "\n")

            count += 1
            # Record and checkpoint the processed id (always append to keep a ledger)
            seen_ids.add(result["doc_id"])
            _append_processed_id(ckpt_path, result["doc_id"])  # checkpoint after each doc
            _update_summary(summary, result)
            if count % 100 == 0:
                print(f"Processed {count} docs...")
        print(f"Done. Processed {count} documents (deduped {dupes} duplicates). Outputs in {out_path}")
    finally:
        steps_f.close(); edges_f.close(); meta_f.close(); ents_f.close()
        if write_summary:
            _finalize_summary(summary, out_path)


def main():
    ap = argparse.ArgumentParser(description="Document-by-document ontology scan")
    ap.add_argument("--index", type=str, default=None, help="Index or alias (defaults to config alias)")
    ap.add_argument("--limit", type=int, default=1000, help="Max docs to process (0 = all)")
    ap.add_argument("--batch", type=int, default=200, help="Batch size for PIT pagination")
    ap.add_argument("--out-dir", type=str, default="outputs/ontology_scan", help="Output directory")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip processed doc_ids)")
    ap.add_argument("--checkpoint-file", type=str, default=None, help="Path to checkpoint ids file (defaults to <out-dir>/checkpoint.ids)")
    ap.add_argument("--no-summary", action="store_true", help="Disable writing summary.json")
    args = ap.parse_args()

    run(
        index=args.index,
        limit=(None if args.limit == 0 else args.limit),
        batch=args.batch,
        out_dir=args.out_dir,
        resume=args.resume,
        checkpoint_file=args.checkpoint_file,
        write_summary=(not args.no_summary),
    )


if __name__ == "__main__":
    main()
