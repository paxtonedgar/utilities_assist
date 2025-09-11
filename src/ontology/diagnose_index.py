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
from typing import Any, Dict, Iterable, List, Tuple, Optional

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


STOPWORDS = set("""
the a an to for of and or in on with by from as at be is are was were can will this that these those into over under about per not no yes it its it's your you we they them his her their our ours has have had do does did done should must may might would could than then next first second finally after when once before via using use used make create configure open click select choose run execute set update enable install remove delete add review approve reject
""".split())


def extract_keywords(sample: str, k: int = 10) -> List[str]:
    # Very light keyword extraction: capitalized terms, backticked tokens, and long words
    import re
    toks = re.findall(r"`([^`]+)`|\b([A-Z][A-Za-z0-9_-]{2,})\b|\b([a-z][a-z0-9_-]{4,})\b", sample)
    words: List[str] = []
    for t in toks:
        cand = next((x for x in t if x), None)
        if not cand:
            continue
        low = cand.lower()
        if low in STOPWORDS:
            continue
        words.append(cand)
    # Deduplicate preserving order
    seen = set(); out = []
    for w in words:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) >= k:
            break
    return out


def _make_preview_writer(base: Path, use_gzip: bool, split_every: int) -> Tuple[Any, callable, callable]:
    """Return (handle, write_fn, close_fn) that writes JSONL, optionally gzipped and split.

    split_every: number of docs per part; 0 = no split.
    """
    import gzip
    part = 0
    count = 0

    def _open_part() -> Any:
        nonlocal part
        part += 1
        name = f"docs_preview.part-{part:05d}.jsonl" if split_every else "docs_preview.jsonl"
        path = base / name
        if use_gzip:
            return gzip.open(str(path) + ".gz", mode="wt", encoding="utf-8")
        return path.open("w", encoding="utf-8")

    fh = _open_part()

    def write(line: str):
        nonlocal count, fh
        fh.write(line)
        if split_every:
            count += 1
            if count % split_every == 0:
                fh.close()
                fh = _open_part()

    def close():
        fh.close()

    return fh, write, close


def diagnose(
    indices: List[str],
    out_dir: Path,
    limit: int = 25,
    batch: int = 200,
    redact: str = "none",
    max_candidates: int = 6,
    min_sample_len: int = 80,
    paths_include: Optional[str] = None,
    paths_exclude: Optional[str] = None,
    gzip_out: bool = False,
    split_every: int = 0,
    summary_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenSearchClient()

    for index in indices:
        base = out_dir / index
        base.mkdir(parents=True, exist_ok=True)

        fields: Dict[str, PathStat] = defaultdict(PathStat)
        if not summary_only:
            _fh, write_line, close_writer = _make_preview_writer(base, gzip_out, split_every)
        else:
            _fh = None
            write_line = lambda _x: None
            close_writer = lambda: None
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
                    if len(val_str) < min_sample_len:
                        continue
                    if len(val_str) > 100000:
                        continue
                    # optional path filters
                    if paths_include:
                        try:
                            import re as _re
                            if not _re.search(paths_include, path):
                                continue
                        except Exception:
                            pass
                    if paths_exclude:
                        try:
                            import re as _re
                            if _re.search(paths_exclude, path):
                                continue
                        except Exception:
                            pass
                    is_html = looks_like_html(val_str)
                    candidates.append((path, val_str, is_html))
                    fields[path].add(val_str, is_html)

            # choose top 6 by length for preview
            top = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[:max_candidates]
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

                item: Dict[str, Any] = {"path": p, "len": len(v), "html": is_html, "patterns": all_counts}
                if redact == "none":
                    item["sample"] = sample[:600]
                elif redact == "hash":
                    import hashlib
                    item["sample_hash"] = hashlib.sha256(sample.encode("utf-8")).hexdigest()
                elif redact == "keywords":
                    item["keywords"] = extract_keywords(sample, k=12)
                # else: no sample included
                preview_items.append(item)
            if not summary_only:
                write_line(
                    json.dumps({"doc_id": doc_id, "index": index, "top_candidates": preview_items})
                    + "\n"
                )

            seen += 1
        close_writer()

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
    ap.add_argument("--redact", type=str, default="none", choices=["none", "hash", "keywords", "nosample"], help="Redact policy for previews")
    ap.add_argument("--max-candidates", type=int, default=6, help="Max top candidate fields per doc in preview")
    ap.add_argument("--min-len", type=int, default=80, help="Minimum content length to consider a candidate field")
    ap.add_argument("--paths-include", type=str, default=None, help="Regex to include only matching field paths")
    ap.add_argument("--paths-exclude", type=str, default=None, help="Regex to exclude field paths")
    ap.add_argument("--gzip", action="store_true", help="Gzip preview files")
    ap.add_argument("--split", type=int, default=0, help="Split preview into parts every N docs (0=no split)")
    ap.add_argument("--summary-only", action="store_true", help="Skip docs_preview; write only summaries")
    args = ap.parse_args()

    if args.indices.lower() in {"khub", "khub*", "auto-khub"}:
        idxs = _discover_khub_indices(prefix="khub")
    else:
        idxs = [s.strip() for s in args.indices.split(",") if s.strip()]

    diagnose(
        idxs,
        out_dir=Path(args.out),
        limit=args.limit,
        batch=args.batch,
        redact=("none" if args.redact == "none" else args.redact),
        max_candidates=args.max_candidates,
        min_sample_len=args.min_len,
        paths_include=args.paths_include,
        paths_exclude=args.paths_exclude,
        gzip_out=bool(args.gzip),
        split_every=args.split,
        summary_only=bool(args.summary_only),
    )


if __name__ == "__main__":
    main()
