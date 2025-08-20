# tools/coverage_cli.py
"""
CLI tool for testing coverage evaluation system.

Usage:
    python tools/coverage_cli.py "onboard AU" < sample_passages.json

Expected input format (JSON):
[
    {
        "text": "Content text...",
        "url": "http://example.com",
        "title": "Document title",
        "heading": "Section heading",
        "rank": 1
    },
    ...
]
"""

import json
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.quality.coverage import CoverageGate, Passage
from src.quality.subquery import decompose


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/coverage_cli.py <query>", file=sys.stderr)
        print("Input: JSON array of passages on stdin", file=sys.stderr)
        sys.exit(1)
    
    query = sys.argv[1]
    
    try:
        # Read passages from stdin
        data = json.load(sys.stdin)
        
        # Convert to Passage objects
        passages = []
        for p in data:
            passages.append(Passage(
                text=p.get("text", ""),
                meta={
                    "url": p.get("url", ""),
                    "title": p.get("title", ""),
                    "heading": p.get("heading", ""),
                    "rank": p.get("rank", 999)
                }
            ))
        
        # Generate subqueries
        subqs = decompose(query)
        
        # Create coverage gate
        gate = CoverageGate()
        
        # Evaluate coverage
        ev = gate.evaluate(query, subqs, passages)
        
        # Output results
        result = {
            "query": query,
            "subqueries": subqs,
            "aspect_recall": round(ev["aspect_recall"], 3),
            "alpha_ndcg": round(ev["alpha_ndcg"], 3),
            "actionable_spans": ev["actionable_spans"],
            "gate_pass": ev["gate_pass"],
            "total_passages": len(passages),
            "selected_passages": sum(len(idxs) for idxs in ev["picks"].values()),
            "picks_per_aspect": {i: len(idxs) for i, idxs in ev["picks"].items()}
        }
        
        print(json.dumps(result, indent=2))
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()