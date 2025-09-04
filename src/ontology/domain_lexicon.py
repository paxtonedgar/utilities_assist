"""
Domain lexicon and regexes for teams/divisions/apps/platforms/diagrams.
"""

from __future__ import annotations
import re
from typing import Set


TEAM_KEYS = re.compile(r"^(team|owner|maintainers?|contact)\s*:\s*(.+)$", re.IGNORECASE)
DIVISION_KEYS = re.compile(r"^(division|org|organization|department)\s*:\s*(.+)$", re.IGNORECASE)
APP_KEYS = re.compile(r"^(application|app|service)\s*:\s*(.+)$", re.IGNORECASE)
TOOL_KEYS = re.compile(r"^(tool|product)\s*:\s*(.+)$", re.IGNORECASE)

PLATFORM_TOKENS: Set[str] = {"eks", "gap", "gks", "gkp"}
PLATFORM_CUES = re.compile(r"\b(EKS|GAP|GKS|GKP)\b", re.IGNORECASE)
DEPLOYMENT_CUES = re.compile(r"\b(kubectl|helm|argo|gkp|deploy|upgrade|promote)\b", re.IGNORECASE)

DIAGRAM_URL = re.compile(r"https?://[^\s)]+\.(?:png|svg|jpg|jpeg|gif|drawio|pdf)", re.IGNORECASE)
DIAGRAM_WORD = re.compile(r"\b(architecture|diagram|topology|design)\b", re.IGNORECASE)

HEADER_CUES = re.compile(
    r"^(Onboarding|Setup|Install|Configure|Deploy|Verify|Troubleshoot|Architecture|Ownership|Team|Division)\b",
    re.IGNORECASE,
)

