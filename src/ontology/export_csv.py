"""
Convert ontology NDJSON outputs to CSV files suited for Neo4j LOAD CSV.

Inputs: a directory containing docs.ndjson, steps.ndjson, edges.ndjson, entities.ndjson
Outputs: CSV files (docs.csv, steps.csv, edges.csv, entities.csv) in --out dir

Design choices:
- Composite IDs to avoid cross-index collisions:
  - Doc.id = "{index}/{doc_id}"
  - Step.id = "{index}/{doc_id}_{order}"
- Keep minimal, flat columns so Cypher LOAD CSV WITH HEADERS is straightforward.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable
import argparse


def _read_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _safe_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "t", "yes", "y")
    if isinstance(v, (int, float)):
        return bool(v)
    return False


def export_csv(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_in = in_dir / "docs.ndjson"
    steps_in = in_dir / "steps.ndjson"
    edges_in = in_dir / "edges.ndjson"
    entities_in = in_dir / "entities.ndjson"

    # Docs
    docs_rows = []
    for d in _read_ndjson(docs_in):
        idx = d.get("index") or d.get("index_name") or ""
        doc_id = d.get("doc_id") or d.get("id") or ""
        comp_id = f"{idx}/{doc_id}" if idx else doc_id
        docs_rows.append(
            {
                "id": comp_id,
                "doc_id": doc_id,
                "index_name": idx,
                "step_cnt": d.get("steps", ""),
                "edge_cnt": d.get("edges", ""),
            }
        )

    if docs_rows:
        with (out_dir / "docs.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "doc_id", "index_name", "step_cnt", "edge_cnt"])
            w.writeheader(); w.writerows(docs_rows)

    # Steps
    steps_rows = []
    for s in _read_ndjson(steps_in):
        idx = s.get("index") or ""
        doc_id = s.get("doc_id") or ""
        order = s.get("order")
        comp_doc = f"{idx}/{doc_id}" if idx else doc_id
        step_id = f"{comp_doc}_{order}"
        steps_rows.append(
            {
                "id": step_id,
                "doc_composite_id": comp_doc,
                "doc_id": doc_id,
                "index_name": idx,
                "label": s.get("canonical_label") or s.get("label"),
                "verb": s.get("verb"),
                "obj": s.get("object") or s.get("obj"),
                "section": s.get("section_title") or s.get("section"),
                "order": s.get("order"),
                "page_url": s.get("page_url"),
            }
        )

    if steps_rows:
        with (out_dir / "steps.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "doc_composite_id",
                    "doc_id",
                    "index_name",
                    "label",
                    "verb",
                    "obj",
                    "section",
                    "order",
                    "page_url",
                ],
            )
            w.writeheader(); w.writerows(steps_rows)

    # Edges (NEXT only)
    edges_rows = []
    for e in _read_ndjson(edges_in):
        typ = _coalesce(e.get("type"), "NEXT")
        if typ != "NEXT":
            continue
        idx = _coalesce(e.get("index"), e.get("a", {}).get("index"), e.get("b", {}).get("index"), "")
        doc_id = e.get("doc_id") or _coalesce(e.get("doc_id_a"), e.get("doc_id_b"), "")
        src_order = _coalesce(e.get("src_order"), e.get("order_a"))
        dst_order = _coalesce(e.get("dst_order"), e.get("order_b"))
        comp_doc = f"{idx}/{doc_id}" if idx else doc_id
        src = f"{comp_doc}_{src_order}"
        dst = f"{comp_doc}_{dst_order}"
        edges_rows.append(
            {
                "src": src,
                "dst": dst,
                "score": e.get("score"),
                "accepted": "true" if _safe_bool(e.get("accepted")) else "false",
            }
        )

    if edges_rows:
        with (out_dir / "edges.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["src", "dst", "score", "accepted"])
            w.writeheader(); w.writerows(edges_rows)

    # Entities (optional)
    entities_rows = []
    for ent in _read_ndjson(entities_in):
        t = ent.get("type"); n = ent.get("name")
        if t and n:
            entities_rows.append({"type": t, "name": n})

    if entities_rows:
        with (out_dir / "entities.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["type", "name"])
            w.writeheader(); w.writerows(entities_rows)

    print("✅ Wrote CSVs to:", out_dir)


def main():
    ap = argparse.ArgumentParser(description="Export ontology NDJSON to CSV for LOAD CSV")
    ap.add_argument("--input", required=True, help="Input directory containing NDJSON files")
    ap.add_argument("--out", required=False, help="Output directory for CSV files (default: <input>/csv)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "csv"
    export_csv(in_dir, out_dir)


if __name__ == "__main__":
    main()

