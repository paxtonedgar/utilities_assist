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


