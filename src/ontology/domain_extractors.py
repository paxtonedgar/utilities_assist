"""
Heuristic extractors for Teams, Divisions, Applications, Tools, Platforms, and Diagram links.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re

from .domain_lexicon import (
    TEAM_KEYS,
    DIVISION_KEYS,
    APP_KEYS,
    TOOL_KEYS,
    PLATFORM_CUES,
    DEPLOYMENT_CUES,
    DIAGRAM_URL,
    DIAGRAM_WORD,
    HEADER_CUES,
)


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def extract_team(line: str) -> str | None:
    m = TEAM_KEYS.match(line)
    return m.group(2).strip() if m else None


def extract_division(line: str) -> str | None:
    m = DIVISION_KEYS.match(line)
    return m.group(2).strip() if m else None


def extract_application(line: str) -> str | None:
    m = APP_KEYS.match(line)
    return m.group(2).strip() if m else None


def extract_tool(line: str) -> str | None:
    m = TOOL_KEYS.match(line)
    return m.group(2).strip() if m else None


def extract_platforms(text: str) -> List[str]:
    plats = set()
    for m in PLATFORM_CUES.finditer(text or ""):
        plats.add(m.group(0).upper())
    return sorted(plats)


def extract_diagrams(text: str) -> List[str]:
    urls = [m.group(0) for m in DIAGRAM_URL.finditer(text or "")]
    return urls


def find_header_cue(text: str) -> float:
    # Simple boost if a header-like cue exists at the beginning of any line
    for ln in _lines(text):
        if HEADER_CUES.match(ln):
            return 1.0
    return 0.0


def extract_domain_entities_and_relations(passage: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract entities (Team, Division, Application, Tool, Platform, Diagram) and candidate relations from a passage.

    Returns (entities, edges) where each is a list of dicts including minimal fields.
    """
    text = passage.get("text", "")
    ents: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    teams: List[str] = []
    divisions: List[str] = []
    apps: List[str] = []
    tools: List[str] = []

    for ln in _lines(text):
        t = extract_team(ln)
        if t:
            teams.append(t)
        d = extract_division(ln)
        if d:
            divisions.append(d)
        a = extract_application(ln)
        if a:
            apps.append(a)
        tl = extract_tool(ln)
        if tl:
            tools.append(tl)

    plats = extract_platforms(text)
    diags = extract_diagrams(text)
    header_cue = find_header_cue(text)

    # Deduplicate while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    teams = _dedup(teams)
    divisions = _dedup(divisions)
    apps = _dedup(apps)
    tools = _dedup(tools)

    # Build entity objects
    for name in teams:
        ents.append({"type": "Team", "name": name})
    for name in divisions:
        ents.append({"type": "Division", "name": name})
    for name in apps:
        ents.append({"type": "Application", "name": name})
    for name in tools:
        ents.append({"type": "Tool", "name": name})
    for name in plats:
        ents.append({"type": "Platform", "name": name})
    for url in diags:
        ents.append({"type": "Diagram", "name": url, "url": url})

    # Relations within the passage context
    # Team IN_DIVISION Division
    for t in teams:
        for d in divisions:
            edges.append(
                {
                    "type": "IN_DIVISION",
                    "a": {"type": "Team", "name": t},
                    "b": {"type": "Division", "name": d},
                    "signals": {"header_cue": header_cue},
                    "evidence_refs": [
                        {"doc_id": passage.get("doc_id"), "snippet": f"Team: {t} | Division: {d}"}
                    ],
                }
            )

    # Team OWNS Application/Tool (ownership cues from explicit keys)
    if teams and (apps or tools):
        for t in teams:
            for a in apps:
                edges.append(
                    {
                        "type": "OWNS",
                        "a": {"type": "Team", "name": t},
                        "b": {"type": "Application", "name": a},
                        "signals": {"ownership_cue": 1.0, "header_cue": header_cue},
                        "evidence_refs": [
                            {"doc_id": passage.get("doc_id"), "snippet": f"Team: {t} | Application: {a}"}
                        ],
                    }
                )
            for tool in tools:
                edges.append(
                    {
                        "type": "OWNS",
                        "a": {"type": "Team", "name": t},
                        "b": {"type": "Tool", "name": tool},
                        "signals": {"ownership_cue": 1.0, "header_cue": header_cue},
                        "evidence_refs": [
                            {"doc_id": passage.get("doc_id"), "snippet": f"Team: {t} | Tool: {tool}"}
                        ],
                    }
                )

    # Application RUNS_ON Platform (platform cues)
    for a in apps:
        for p in plats:
            edges.append(
                {
                    "type": "RUNS_ON",
                    "a": {"type": "Application", "name": a},
                    "b": {"type": "Platform", "name": p.upper()},
                    "signals": {"platform_cue": 1.0},
                    "evidence_refs": [
                        {"doc_id": passage.get("doc_id"), "snippet": f"{a} on {p}"}
                    ],
                }
            )

    # Diagram DOCUMENTED_BY Team/Application
    for url in diags:
        for t in teams:
            edges.append(
                {
                    "type": "DOCUMENTED_BY",
                    "a": {"type": "Team", "name": t},
                    "b": {"type": "Diagram", "name": url, "url": url},
                    "signals": {"diagram_cue": 1.0},
                    "evidence_refs": [
                        {"doc_id": passage.get("doc_id"), "snippet": url}
                    ],
                }
            )
        for a in apps:
            edges.append(
                {
                    "type": "DOCUMENTED_BY",
                    "a": {"type": "Application", "name": a},
                    "b": {"type": "Diagram", "name": url, "url": url},
                    "signals": {"diagram_cue": 1.0},
                    "evidence_refs": [
                        {"doc_id": passage.get("doc_id"), "snippet": url}
                    ],
                }
            )

    return ents, edges

