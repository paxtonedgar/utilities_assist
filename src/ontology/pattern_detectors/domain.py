from __future__ import annotations

from typing import Dict, List
import re


DEFAULT_DOMAIN_TERMS: List[str] = [
    # seed with a few placeholder utilities/systems; customize via config later
    "ServiceNow", "Splunk", "Jira", "Confluence", "CIU", "ETU", "PCU",
]


def analyze(text: str, extra_terms: List[str] | None = None) -> Dict[str, int]:
    terms = (extra_terms or []) + DEFAULT_DOMAIN_TERMS
    hits = 0
    for t in terms:
        try:
            if re.search(rf"\b{re.escape(t)}\b", text):
                hits += 1
        except Exception:
            pass
    return {"domain_terms": hits}

