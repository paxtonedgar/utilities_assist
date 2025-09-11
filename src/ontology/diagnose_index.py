"""
Ontology diagnostics: explore index field structures and sample content.

Purpose:
- Sample documents from one or more indices
- Recursively enumerate field paths and text-like values
- Detect likely-HTML vs plain text and provide text previews
- Emit summaries per index to outputs/diagnostics/<index>/ for inspection

Usage examples:
  python -m src.ontology.diagnose_index --indices khub --limit 25
  python -m src.ontology.diagnose_index --indices khub-opensearch-swagger-index --limit 50

Outputs per index:
- fields_summary.json  (path stats: count, html_count, avg_len, examples)
- docs_preview.jsonl    (per-doc: doc_id, top candidate fields with previews)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
from src.infra.opensearch_client import OpenSearchClient


HTML_TAG_RE = re.compile(r"<[^>]+>")

# Pattern detectors
from .pattern_detectors import structural as det_struct
from .pattern_detectors import linguistic as det_ling
from .pattern_detectors import confluence as det_conf
from .pattern_detectors import domain as det_domain


def looks_like_html(s: str) -> bool:
    if "<" in s and ">" in s:
        # quick check for common tags
        return bool(re.search(r"</?\w+[^>]*>", s))
    return False


def strip_html(s: str) -> str:
    try:
        # lightweight strip; avoids adding dependencies
        return HTML_TAG_RE.sub(" ", s)
    except Exception:
        return s


def _path_join(base: str, key: str) -> str:
    if not base:
        return key
    return f"{base}.{key}"


def iter_paths(obj: Any, base: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (path, value) pairs for all leaves in a nested object.

    Lists are represented with '[]' at that segment, e.g., 'sections[].content'.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_paths(v, _path_join(base, k))
    elif isinstance(obj, list):
        # use a placeholder for list segment
        list_base = (base + "[]") if base else "[]"
        for it in obj[:5]:  # limit breadth
            yield from iter_paths(it, list_base)
    else:
        yield base, obj


@dataclass
class PathStat:
    count: int = 0
    html_count: int = 0
    total_len: int = 0
    examples: List[str] = None

    def add(self, value: str, is_html: bool):
        self.count += 1
        self.total_len += len(value)
        if is_html:
            self.html_count += 1
        if self.examples is None:
            self.examples = []
        if len(self.examples) < 3:
            self.examples.append(value[:400])

    def to_dict(self) -> Dict[str, Any]:
        avg_len = (self.total_len / self.count) if self.count else 0
        return {
            "count": self.count,
            "html_count": self.html_count,
            "avg_len": round(avg_len, 1),
            "examples": self.examples or [],
        }


def _discover_khub_indices(prefix: str = "khub") -> List[str]:
    try:
        import requests

        s = get_settings()
        base = s.opensearch_host.rstrip("/")
        _setup_jpmc_proxy()
        auth = _get_aws_auth()
        url = f"{base}/_cat/indices?format=json&expand_wildcards=all"
        r = requests.get(url, auth=auth, timeout=30)
        r.raise_for_status()
        arr = r.json()
        names = [row.get("index") for row in arr if isinstance(row, dict)]
        out = []
        for n in names:
            if not n or n.startswith("."):
                continue
            if n.lower().startswith(prefix.lower()):
                out.append(n)
        return sorted({*out})
    except Exception:
        # fallback to configured alias
        s = get_settings()
        return [s.search_index_alias]


def _score_field(stat: "PathStat", patterns: Dict[str, int]) -> Dict[str, float]:
    # Content density: crude ratio of non-tag text to length proxy
    avg_len = (stat.total_len / max(stat.count, 1))
    structure_score = sum(
        patterns.get(k, 0)
        for k in ["tables", "lists", "ordered_lists", "headers", "code_blocks"]
    )
    extraction_potential = (
        min(avg_len / 1000.0, 2.0)
        + 0.5 * structure_score
        + 0.5 * (patterns.get("proc_cues", 0) + patterns.get("trigger_cues", 0))
    )
    return {
        "avg_len": avg_len,
        "structure_score": structure_score,
        "extraction_potential": round(float(extraction_potential), 2),
    }


def diagnose(indices: List[str], out_dir: Path, limit: int = 25, batch: int = 200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenSearchClient()

    for index in indices:
        base = out_dir / index
        base.mkdir(parents=True, exist_ok=True)

        fields: Dict[str, PathStat] = defaultdict(PathStat)
        docs_preview = (base / "docs_preview.jsonl").open("w", encoding="utf-8")
        pattern_freq = Counter()
        opportunities: List[Dict[str, Any]] = []

        seen = 0
        for hit in client.iterate_index(index=index, fields=None, batch_size=batch, max_docs=limit):
            doc_id = hit.get("_id")
            src = hit.get("_source", {})

            # gather candidates: text-like leaves
            candidates: List[Tuple[str, str, bool]] = []  # (path, value, is_html)
            for path, val in iter_paths(src):
                if not path:
                    continue
                if isinstance(val, str):
                    val_str = val.strip()
                    if len(val_str) < 20:
                        continue
                    if len(val_str) > 100000:
                        continue
                    is_html = looks_like_html(val_str)
                    candidates.append((path, val_str, is_html))
                    fields[path].add(val_str, is_html)

            # choose top 6 by length for preview
            top = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[:6]
            preview_items = []
            for p, v, is_html in top:
                sample = (strip_html(v) if is_html else v)
                # run pattern detectors
                s_counts = det_struct.analyze(v, is_html)
                l_counts = det_ling.analyze(sample)
                c_counts = det_conf.analyze_html(v) if is_html else {}
                d_counts = det_domain.analyze(sample)
                all_counts = {**s_counts, **l_counts, **c_counts, **d_counts}
                pattern_freq.update(all_counts)

                preview_items.append(
                    {
                        "path": p,
                        "len": len(v),
                        "html": is_html,
                        "sample": sample[:600],
                        "patterns": all_counts,
                    }
                )
            docs_preview.write(
                json.dumps({"doc_id": doc_id, "index": index, "top_candidates": preview_items})
                + "\n"
            )

            seen += 1
        docs_preview.close()

        # write summary
        # compute simple field-level opportunities from accumulated stats
        field_opps: List[Dict[str, Any]] = []
        for p, stat in fields.items():
            patterns = {}  # we don't track per-path pattern sums; estimate from examples
            score = _score_field(stat, patterns)
            field_opps.append({
                "path": p,
                "count": stat.count,
                "avg_len": score["avg_len"],
                "extraction_potential": score["extraction_potential"],
            })

        # sort opportunities
        field_opps_sorted = sorted(field_opps, key=lambda x: (-x["extraction_potential"], -x["avg_len"]))

        summary = {
            "index": index,
            "documents_sampled": seen,
            "paths": {p: stat.to_dict() for p, stat in sorted(fields.items())},
            "pattern_frequency": dict(pattern_freq),
        }
        with (base / "fields_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # write aggregate pattern frequency
        with (base / "pattern_frequency.json").open("w", encoding="utf-8") as f:
            json.dump(dict(pattern_freq), f, indent=2)

        # write CSV of extraction opportunities
        import csv
        with (base / "extraction_opportunities.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "count", "avg_len", "extraction_potential"])
            w.writeheader(); w.writerows(field_opps_sorted)

        print(f"✅ Wrote diagnostics for {index}: {base}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose index content fields and HTML")
    ap.add_argument("--indices", type=str, default="khub", help="Comma-separated list or 'khub' to auto-discover khub* indices")
    ap.add_argument("--limit", type=int, default=25, help="Docs per index to sample")
    ap.add_argument("--out", type=str, default="outputs/diagnostics", help="Output directory base")
    ap.add_argument("--batch", type=int, default=200, help="Batch size for iteration")
    args = ap.parse_args()

    if args.indices.lower() in {"khub", "khub*", "auto-khub"}:
        idxs = _discover_khub_indices(prefix="khub")
    else:
        idxs = [s.strip() for s in args.indices.split(",") if s.strip()]

    diagnose(idxs, out_dir=Path(args.out), limit=args.limit, batch=args.batch)


if __name__ == "__main__":
    main()
