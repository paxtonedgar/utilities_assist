import json, sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from quality.coverage import CoverageGate, Passage
from quality.subquery import decompose

if __name__ == "__main__":
    q = sys.argv[1]
    data = json.load(sys.stdin)  # expects [{"text": "...","url":"...","title":"...","heading":"...","rank":1}, ...]
    passages = [Passage(p["text"], {"url": p.get("url",""), "title": p.get("title",""),
                                    "heading": p.get("heading",""), "rank": p.get("rank", 9999)}) for p in data]
    subqs = decompose(q)
    gate = CoverageGate()
    ev = gate.evaluate(q, subqs, passages)
    print(json.dumps({
        "subqs": subqs,
        "aspect_recall": ev["aspect_recall"],
        "alpha_ndcg": ev["alpha_ndcg"],
        "actionable_spans": ev["actionable_spans"],
        "gate_pass": ev["gate_pass"]
    }, indent=2))