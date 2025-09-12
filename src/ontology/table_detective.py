from __future__ import annotations

"""
Table Detective: hunt down tables across formats to diagnose extraction gaps.

Usage:
  python -m src.ontology.table_detective --index khub-product-apg-index --limit 10
  python -m src.ontology.table_detective --diag-dir outputs/diagnostics/khub-product-apg-index --index khub-product-apg-index --limit 10
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.infra.opensearch_client import OpenSearchClient
from .structure_parser import hunt_for_tables, _parse_markdown_tables_enhanced, _parse_confluence_tables, _parse_html


def _load_table_doc_ids(diag_dir: Path) -> List[str]:
    ids: List[str] = []
    p = diag_dir / "docs_preview.jsonl"
    if not p.exists():
        return ids
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                tops = obj.get("top_candidates") or []
                if any((c.get("patterns") or {}).get("tables", 0) > 0 for c in tops):
                    ids.append(obj.get("doc_id"))
            except Exception:
                continue
    return ids


def investigate_tables(index: str, diag_dir: str | None, limit: int) -> None:
    client = OpenSearchClient()
    doc_ids: List[str] = []
    if diag_dir:
        doc_ids = _load_table_doc_ids(Path(diag_dir))[:limit]

    def _yield_hits():
        if doc_ids:
            for did in doc_ids:
                try:
                    yield client.get_doc_by_id(index=index, doc_id=did)
                except Exception:
                    continue
        else:
            # sample first N via iterator
            count = 0
            for hit in client.iterate_index(index=index, fields=None, batch_size=200, max_docs=limit):
                yield hit
                count += 1
                if count >= limit:
                    break

    for hit in _yield_hits():
        doc_id = hit.get("_id")
        findings = hunt_for_tables(hit)
        if findings:
            print(f"\nDoc {doc_id}: Found {len(findings)} table indicators")
            for f in findings[:3]:
                print(f"  - {f}")
        # Try parsers on sections
        src = hit.get("_source", {})
        sections = src.get("sections") if isinstance(src.get("sections"), list) else []
        for s in sections[:3]:
            content = (s.get("content") or s.get("body") or s.get("text") or "")
            if not isinstance(content, str) or len(content) < 20:
                continue
            html_tables = []
            if "<" in content and "</" in content:
                html_tables = _parse_html(content).get("tables", [])
            md_tables = _parse_markdown_tables_enhanced(content)
            conf_tables = _parse_confluence_tables(content)
            if html_tables:
                print(f"  HTML tables: {len(html_tables)} (section: {s.get('heading') or s.get('title')})")
            if md_tables:
                print(f"  Markdown tables: {len(md_tables)} (section: {s.get('heading') or s.get('title')})")
            if conf_tables:
                print(f"  Confluence tables: {len(conf_tables)} (section: {s.get('heading') or s.get('title')})")


def main():
    ap = argparse.ArgumentParser(description="Investigate table formats in an index")
    ap.add_argument("--index", type=str, required=True, help="Index name or alias")
    ap.add_argument("--limit", type=int, default=10, help="Docs to inspect")
    ap.add_argument("--diag-dir", type=str, default=None, help="Diagnostics directory to sample known table docs")
    args = ap.parse_args()
    investigate_tables(args.index, args.diag_dir, args.limit)


if __name__ == "__main__":
    main()

