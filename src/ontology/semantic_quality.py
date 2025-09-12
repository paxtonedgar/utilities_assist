from __future__ import annotations

"""
Phase 1 Quality Assessment

Reads semantic map outputs (topics.json, doc_map.jsonl), optionally diagnostics
(fields_summary.json, pattern_priority.json), and prints a concise quality
report. Also writes machine‑readable metrics.json/quality_report.txt.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter, defaultdict


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def assess_topics(topics: Dict[str, Any], doc_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "total_topics": 0,
        "labeled_topics": 0,
        "high_confidence_topics": 0,
        "topic_sizes": [],
        "label_distribution": {},
    }
    if not topics:
        return metrics

    # sizes from doc_map
    topic_counts = Counter(int(d.get("topic_id", -1)) for d in doc_map)
    metrics["total_topics"] = len(topics)
    metrics["topic_sizes"] = sorted(topic_counts.values(), reverse=True)

    labeled = 0
    high = 0
    label_hist = Counter()
    for tid, card in topics.items():
        lbl = (card or {}).get("label")
        if lbl:
            labeled += 1
            label_hist[str(lbl)] += 1
        conf = float((card or {}).get("confidence", 0.0) or 0.0)
        if conf >= 0.7:
            high += 1
    metrics["labeled_topics"] = labeled
    metrics["high_confidence_topics"] = high
    metrics["label_distribution"] = dict(label_hist)
    return metrics


def assess_segments(doc_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total_docs": len(doc_map),
        "docs_with_segments": 0,
        "avg_segments_per_doc": 0.0,
        "segment_type_distribution": {},
        "table_classification_distribution": {},
        "label_confidence_avg": 0.0,
        "high_confidence_docs": 0,
        "empty_segment_rate": 0.0,
    }
    if not doc_map:
        return out

    seg_counts: List[int] = []
    seg_types = Counter()
    table_types = Counter()
    confs: List[float] = []

    for d in doc_map:
        segs = d.get("segments") or []
        seg_counts.append(len(segs))
        if segs:
            out["docs_with_segments"] += 1
        for s in segs:
            seg_types[str(s.get("type"))] += 1
            if s.get("type") == "Table":
                table_types[str(s.get("table_type", "unknown"))] += 1
        confs.append(float(d.get("label_confidence", 0.0) or 0.0))

    n = len(doc_map)
    out["avg_segments_per_doc"] = round(sum(seg_counts) / max(n, 1), 3)
    out["empty_segment_rate"] = round((n - out["docs_with_segments"]) / max(n, 1), 3)
    out["segment_type_distribution"] = dict(seg_types)
    out["table_classification_distribution"] = dict(table_types)
    out["label_confidence_avg"] = round(sum(confs) / max(n, 1), 3)
    out["high_confidence_docs"] = sum(1 for c in confs if c >= 0.7)
    return out


def compare_with_diagnostics(semantic_dir: Path, diagnostics_dir: Path) -> Dict[str, Any]:
    """Compare Phase 1 outputs against diagnostics expectations when available."""
    result: Dict[str, Any] = {}
    fields_summary = _read_json(diagnostics_dir / "fields_summary.json")
    pattern_priority = _read_json(diagnostics_dir / "pattern_priority.json")
    doc_map = _read_jsonl(semantic_dir / "doc_map.jsonl")

    # Diagnostics tables expectation
    expected_tables = 0
    try:
        # crude estimate: count of 'tables' occurrences in pattern_frequency / per-doc rate * N
        pf = fields_summary.get("pattern_frequency") or {}
        expected_tables = int(pf.get("tables", 0))
    except Exception:
        pass
    found_tables = sum(1 for d in doc_map for s in (d.get("segments") or []) if s.get("type") == "Table")
    result["tables_expected"] = expected_tables
    result["tables_found"] = found_tables

    # Diagnostics proc_cues per doc vs StepBlocks found
    proc_rate = float((pattern_priority.get("per_doc_rate") or {}).get("proc_cues", 0.0))
    N = len(doc_map)
    expected_proc_like = int(round(proc_rate * N))
    found_stepblocks = sum(1 for d in doc_map for s in (d.get("segments") or []) if s.get("type") == "StepBlock")
    result["proc_expected_docs"] = expected_proc_like
    result["stepblock_found_docs"] = found_stepblocks

    return result


def generate_quality_report(semantic_dir: Path, diagnostics_dir: Path | None) -> Dict[str, Any]:
    topics = _read_json(semantic_dir / "topics.json")
    doc_map = _read_jsonl(semantic_dir / "doc_map.jsonl")

    topic_metrics = assess_topics(topics, doc_map)
    segment_metrics = assess_segments(doc_map)
    compare_metrics: Dict[str, Any] = {}
    if diagnostics_dir and (diagnostics_dir / "fields_summary.json").exists():
        compare_metrics = compare_with_diagnostics(semantic_dir, diagnostics_dir)

    # Red flags
    red_flags: List[str] = []
    if segment_metrics["empty_segment_rate"] > 0.5:
        red_flags.append(
            f"High empty segment rate: {segment_metrics['empty_segment_rate']:.2f} (>0.5)"
        )
    tables = segment_metrics.get("table_classification_distribution", {})
    total_tables = sum(tables.values())
    if total_tables > 0 and (tables.get("other", 0) / total_tables) > 0.4:
        red_flags.append(">40% of tables classified as 'other'")
    if topic_metrics.get("labeled_topics", 0) == 0:
        red_flags.append("No topics labeled — LLM labeling likely failed")

    return {
        "topics": topic_metrics,
        "segments": segment_metrics,
        "compare": compare_metrics,
        "red_flags": red_flags,
    }


def _print_report(m: Dict[str, Any]) -> str:
    lines: List[str] = []
    topics = m.get("topics", {})
    segs = m.get("segments", {})
    comp = m.get("compare", {})

    lines.append(f"Total documents: {segs.get('total_docs', 0)}")
    lines.append(f"Total topics: {topics.get('total_topics', 0)} (labeled: {topics.get('labeled_topics', 0)}, high_conf: {topics.get('high_confidence_topics', 0)})")
    if topics.get("label_distribution"):
        lines.append("Label distribution:")
        for k, v in topics["label_distribution"].items():
            lines.append(f"  {k}: {v}")
    lines.append(f"Docs with segments: {segs.get('docs_with_segments', 0)}/{segs.get('total_docs', 0)}")
    lines.append(f"Avg segments/doc: {segs.get('avg_segments_per_doc', 0.0)}")
    if segs.get("segment_type_distribution"):
        lines.append("Segment types:")
        for k, v in sorted(segs["segment_type_distribution"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k}: {v}")
    if segs.get("table_classification_distribution"):
        lines.append("Table classifications:")
        for k, v in segs["table_classification_distribution"].items():
            lines.append(f"  {k}: {v}")
    lines.append(f"Avg label confidence: {segs.get('label_confidence_avg', 0.0)} (high: {segs.get('high_confidence_docs', 0)})")
    if comp:
        lines.append("Diagnostics comparison:")
        for k, v in comp.items():
            lines.append(f"  {k}: {v}")
    if m.get("red_flags"):
        lines.append("Red flags:")
        for rf in m["red_flags"]:
            lines.append(f"  - {rf}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Quality Reporter")
    ap.add_argument("--semantic-dir", type=str, required=True, help="Semantic map output directory for an index")
    ap.add_argument("--diagnostics-dir", type=str, default=None, help="Diagnostics directory for the same index (optional)")
    args = ap.parse_args()

    semantic_dir = Path(args.semantic_dir)
    diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else None
    metrics = generate_quality_report(semantic_dir, diagnostics_dir)

    # Write outputs
    (semantic_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = _print_report(metrics)
    (semantic_dir / "quality_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

