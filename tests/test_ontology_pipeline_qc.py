import pytest

from src.ontology.pipeline import build_next_candidates, score_edges
from src.ontology.qc import qc_report


def _mk_step(label: str, order: int, doc_id: str = "doc1"):
    return {
        "label": label,
        "verb": label.split()[0].lower(),
        "obj": " ".join(label.split()[1:]) or None,
        "order": order,
        "source": "regex",
        "evidence": {"text_snippet": label},
        "doc_id": doc_id,
    }


def test_next_edges_accept_and_qc_passes():
    steps_by_doc = {
        "doc1": [
            _mk_step("Install tools", 1),
            _mk_step("Configure pipeline", 2),
            _mk_step("Verify deployment", 3),
        ]
    }
    edges = build_next_candidates(steps_by_doc)
    assert len(edges) == 2
    scored = score_edges(edges)
    # Expect acceptance with default thresholds
    assert all(s.get("accepted") for s in scored)
    res = qc_report(edges, scored)
    assert res.violations == []


def test_next_edges_reverse_order_flagged():
    steps_by_doc = {
        "doc1": [
            _mk_step("Install tools", 2),
            _mk_step("Configure pipeline", 1),
        ]
    }
    edges = build_next_candidates(steps_by_doc)
    # build_next_candidates sorts by order, so it will produce 1->2 correctly;
    # simulate a bad edge to test QC
    bad_edge = {
        **edges[0],
        "order_a": 3,
        "order_b": 1,
    }
    scored = score_edges([bad_edge])
    res = qc_report([bad_edge], scored)
    assert any("NEXT_reverse_order" in v for v in res.violations)

