# src/quality/subquery.py
import regex as re
from typing import List

_PROCEDURE_HINTS = [
    ("prerequisites", ["prerequisite", "requirement", "before you", "pre-req"]),
    (
        "jira",
        ["jira", "servicenow", "request", "intake", "ticket", "form", "project key"],
    ),
    ("owner", ["owner", "team", "contact", "dl", "email", "slack", "support"]),
    ("endpoints", ["endpoint", "api", "path", "url", "route"]),
    ("steps", ["step", "how to", "onboard", "enable", "configure", "setup"]),
    ("sla", ["sla", "timeline", "turnaround"]),
]


def decompose(user_query: str, max_subqs: int = 6) -> List[str]:
    q = user_query.strip().lower()
    subqs = []

    # explicit verbs produce a direct subquery
    if re.search(r"\b(onboard|enable|configure|setup|migrate|create|request)\b", q):
        subqs.append(f"What are the exact steps to {q}?")

    # generate aspect-driven subqueries
    for name, terms in _PROCEDURE_HINTS:
        subqs.append(f"{name.title()} for: {q}")

    # de-dupe / cap
    seen = set()
    out = []
    for s in subqs:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= max_subqs:
            break
    return out or [q]
