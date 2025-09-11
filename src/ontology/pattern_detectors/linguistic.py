from __future__ import annotations

import re
from typing import Dict

# Lightweight keyword/phrase detectors (regexes; keep fast and dependency-free)
PROCEDURAL = [
    r"\bstep\s*\d+\b",
    r"\bfirst\b",
    r"\bsecond\b",
    r"\bthen\b",
    r"\bnext\b",
    r"\bfinally\b",
    r"\bdo\s+the\s+following\b",
    r"\bto\s+\w+\b",  # 'To configure ...'
]

OWNERSHIP = [
    r"\bmanaged\s+by\b",
    r"\bowned\s+by\b",
    r"\bresponsible\s+for\b",
    r"\bmaintain(s|ed)?\b",
]

TRIGGERS = [
    r"\bwhen\b",
    r"\bafter\b",
    r"\bonce\b",
    r"\btriggers?\b",
    r"\binitiates?\b",
]

MENTIONS = [
    r"@\w+",
    r"\[~[\w.\-]+\]",
]

CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_\-]{2,}\b")


def _count_patterns(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.I))


def analyze(text: str) -> Dict[str, int]:
    return {
        "proc_cues": _count_patterns(text, PROCEDURAL),
        "ownership_cues": _count_patterns(text, OWNERSHIP),
        "trigger_cues": _count_patterns(text, TRIGGERS),
        "mentions": _count_patterns(text, MENTIONS),
        "cap_terms": len(CAPITALIZED_TOKEN_RE.findall(text[:2000])),  # cap at 2k chars
    }

