"""
Quality checks and reports for step/entity linkages prior to persistence.

Includes invariants (evidence presence, intra-doc NEXT rules), sampling helpers,
and simple CSV/JSON reporting with signal breakdowns for audit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import csv


@dataclass
class QCResult:
    total_edges: int
    accepted_edges: int
    rejected_edges: int
    violations: List[str]


def check_evidence_present(edge: Dict[str, Any]) -> Tuple[bool, str | None]:
    evid = edge.get("evidence_refs") or []
    ok = bool(evid)
    return ok, None if ok else "missing_evidence"


def check_next_invariants(edge: Dict[str, Any]) -> Tuple[bool, str | None]:
    if edge.get("type") != "NEXT":
        return True, None
    a_doc = edge.get("doc_id_a")
    b_doc = edge.get("doc_id_b")
    if a_doc != b_doc:
        return False, "NEXT_cross_doc"
    # Monotonic order if available
    a_ord = edge.get("order_a")
    b_ord = edge.get("order_b")
    if a_ord is not None and b_ord is not None and b_ord < a_ord:
        return False, "NEXT_reverse_order"
    return True, None


def run_invariants(edges: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    violations: List[Tuple[int, str]] = []
    for i, e in enumerate(edges):
        for check in (check_evidence_present, check_next_invariants):
            ok, reason = check(e)
            if not ok and reason:
                violations.append((i, reason))
    return violations


def summarize_scores(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_rel: Dict[str, Dict[str, float]] = {}
    for s in scored:
        rel = s.get("relation", "?")
        bucket = by_rel.setdefault(rel, {"n": 0, "accepted": 0})
        bucket["n"] += 1
        if s.get("accepted"):
            bucket["accepted"] += 1
    return by_rel


def export_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def qc_report(
    edges: List[Dict[str, Any]],
    scored_edges: List[Dict[str, Any]],
    csv_out: str | None = None,
) -> QCResult:
    violations = run_invariants(edges)
    summary = summarize_scores(scored_edges)

    rows: List[Dict[str, Any]] = []
    for e, s in zip(edges, scored_edges):
        row = {
            "type": e.get("type"),
            "doc_id_a": e.get("doc_id_a"),
            "doc_id_b": e.get("doc_id_b"),
            "order_a": e.get("order_a"),
            "order_b": e.get("order_b"),
            "score": s.get("score"),
            "accepted": s.get("accepted"),
            "notes": ",".join(s.get("notes", [])),
        }
        # unpack a few common signals for convenience
        signals = s.get("signals", {})
        for k in (
            "regex_step",
            "spacy_imperative",
            "bullet_order_consistency",
            "cue_word",
            "same_entity_overlap",
            "embed_sim",
            "nli_entailment",
        ):
            if k in signals:
                row[k] = signals[k]
        rows.append(row)

    if csv_out:
        export_csv(csv_out, rows)

    accepted = sum(1 for s in scored_edges if s.get("accepted"))
    rejected = len(scored_edges) - accepted
    return QCResult(
        total_edges=len(scored_edges),
        accepted_edges=accepted,
        rejected_edges=rejected,
        violations=[f"{i}:{r}" for i, r in violations],
    )

