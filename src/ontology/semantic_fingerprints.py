from __future__ import annotations

"""
Semantic fingerprint builder for Stage 1A.

Builds privacy-safe document strings from diagnostics (keywords + patterns + cleanness)
that can be embedded and clustered (BERTopic/TF‑IDF + KMeans).
"""

from pathlib import Path
from typing import Dict, Any, List
import json

TRACKED_PATTERNS = [
    "lists",
    "ordered_lists",
    "tables",
    "headers",
    "code_blocks",
    "inline_backticks",
    "proc_cues",
    "trigger_cues",
    "cap_terms",
    "macros",
]


def _tokenize_keywords(sample: str | List[str], k: int = 10) -> List[str]:
    if isinstance(sample, list):
        return [f"keyword:{w}" for w in sample[:k]]
    # When docs_preview contains raw sample text, we do not emit raw text; we skip
    return []


def build_fingerprints(diag_dir: Path, max_candidates: int = 4) -> List[Dict[str, Any]]:
    """Create per-doc fingerprints from diagnostics outputs.

    Inputs expected in diag_dir:
      - docs_preview.jsonl (or split/gz variants) with top_candidates (keywords or sample + patterns)
      - index_summary.json (for profile/capacity hints) – optional

    Output: list of {doc_id, index, tokens: List[str], features: {...}}
    """
    docs: List[Dict[str, Any]] = []
    # Load optional index summary for profile hints
    profile_hint = ""
    try:
        idx = json.loads((diag_dir / "index_summary.json").read_text("utf-8"))
        profile_hint = idx.get("profile") or ""
    except Exception:
        pass

    # Support split parts if present
    preview_files = []
    for name in ["docs_preview.jsonl"] + [f.name for f in diag_dir.glob("docs_preview.part-*.jsonl*")]:
        p = diag_dir / name
        if p.exists():
            preview_files.append(p)
    if not preview_files:
        return docs

    import gzip
    import re

    def _open_any(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    for pf in preview_files:
        with _open_any(pf) as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                doc_id = obj.get("doc_id")
                index = obj.get("index")
                candidates = (obj.get("top_candidates") or [])[:max_candidates]

                # Aggregate patterns across candidates
                pat_sum: Dict[str, int] = {k: 0 for k in TRACKED_PATTERNS}
                tokens: List[str] = []
                for c in candidates:
                    # Keywords are preferred for privacy; otherwise skip samples
                    kws = c.get("keywords")
                    if kws:
                        tokens.extend(_tokenize_keywords(kws, k=10))
                    # Patterns
                    pats = c.get("patterns") or {}
                    for k in TRACKED_PATTERNS:
                        pat_sum[k] += int(pats.get(k, 0) or 0)

                # Encode pattern presence/strength as tokens (light weighting)
                for k, v in pat_sum.items():
                    if v <= 0:
                        continue
                    repeat = 1
                    if v >= 2:
                        repeat = 2
                    if v >= 5:
                        repeat = 3
                    tokens.extend([f"pattern:{k}"] * repeat)

                if profile_hint:
                    tokens.append(f"profile:{profile_hint}")

                if not tokens:
                    continue

                docs.append({
                    "doc_id": doc_id,
                    "index": index,
                    "tokens": tokens,
                    "features": {"patterns": pat_sum, "profile": profile_hint},
                })
    return docs

