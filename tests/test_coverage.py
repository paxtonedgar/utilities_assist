import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from quality.coverage import CoverageGate, Passage
from quality.subquery import decompose

def test_aspect_recall_basic():
    gate = CoverageGate()
    q = "enable Payment API"
    subqs = decompose(q, max_subqs=4)
    passages = [
        Passage("Steps:\n1. Create Jira\n2. Get API key\nContact: dl-payments@x", {"rank":1}),
        Passage("GET /v1/payments\nAuth: Bearer ...", {"rank":2}),
        Passage("Owner: Payments Core Team\nSlack #payments", {"rank":3}),
    ]
    ev = gate.evaluate(q, subqs, passages)
    assert ev["aspect_recall"] >= 0.5
    assert ev["actionable_spans"] >= 2