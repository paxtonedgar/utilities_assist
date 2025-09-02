"""Schemas and simple data containers for OpenAI planner/composer.

These are implementation-agnostic so LangGraph nodes can depend on a
stable contract without knowing about OpenAI specifics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Plan:
    """Planner output contract.

    aspects: dynamic list of section names the composer should build.
    filters: search filters that apply to all aspects (e.g., utility_name, content_type).
    k_per_aspect: number of passages to fetch per aspect (small, focused).
    budgets: character budgets per section (renderers may format to Markdown later).
    """

    aspects: List[str] = field(default_factory=lambda: ["overview", "steps", "api"])
    filters: Dict[str, str] = field(default_factory=dict)
    k_per_aspect: int = 3  # helpful over concise; fetch a bit more per aspect
    budgets: Dict[str, int] = field(
        default_factory=lambda: {
            "overview_chars": 500,     # +~40% vs tight 350
            "steps_chars": 900,        # +~50% vs tight 600
            "api_chars": 500,          # +~40% vs tight 350
            "troubleshoot_chars": 600,
        }
    )


@dataclass
class Citation:
    title: str
    url: str


@dataclass
class Step:
    n: int
    text: str
    citation: Citation


@dataclass
class ApiItem:
    name: str
    url: str
    citation: Citation


@dataclass
class Card:
    """Structured composer output.

    Kept JSON-like to be easily serializable and cached.
    """

    utility: Optional[str] = None
    overview: Optional[Dict[str, object]] = None  # {text: str, citations: [Citation]}
    onboarding_steps: List[Step] = field(default_factory=list)
    apis: List[ApiItem] = field(default_factory=list)
    environments: List[Dict[str, object]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    unknown_fields: List[str] = field(default_factory=list)

