# src/retrieval/suites/schemas.py
"""
Data structures for suite manifest system.
Defines schemas for YAML-based tool suite detection patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Pattern
import re


@dataclass
class ExtractorConfig:
    """Configuration for a single pattern extractor within a suite."""

    name: str  # e.g., "ticket_create"
    capability: str  # e.g., "ticket.create"
    weight: float  # confidence multiplier (0.1-3.0)
    patterns: List[str]  # regex patterns as strings
    attrs: Dict[str, str] = field(default_factory=dict)  # named capture groups
    url_patterns: List[str] = field(default_factory=list)  # URL extraction patterns

    # Compiled patterns (populated after loading)
    _compiled_patterns: List[Pattern] = field(default_factory=list, init=False)
    _compiled_url_patterns: List[Pattern] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Compile regex patterns for performance."""
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.patterns
        ]
        self._compiled_url_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.url_patterns
        ]


@dataclass
class SuiteManifest:
    """Complete manifest for a tool suite (loaded from YAML)."""

    suite: str  # e.g., "jira", "servicenow"
    hints: List[str] = field(default_factory=list)  # fast suite detection patterns
    extractors: List[ExtractorConfig] = field(
        default_factory=list
    )  # pattern extractors

    # Compiled hint patterns (populated after loading)
    _compiled_hints: List[Pattern] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Compile hint patterns for performance."""
        self._compiled_hints = [re.compile(hint, re.IGNORECASE) for hint in self.hints]

    def score_hints(self, passages: List["Passage"]) -> float:
        """
        Quick suite confidence score based on hint patterns.

        Args:
            passages: List of document passages to scan

        Returns:
            Confidence score 0.0-1.0 for this suite being relevant
        """
        if not self._compiled_hints:
            return 0.5  # Default confidence if no hints

        total_matches = 0
        total_possible = len(self._compiled_hints) * len(passages)

        for passage in passages:
            # Check title, URL, and text snippets
            text_to_check = f"{passage.title} {passage.url} {passage.text[:500]}"

            for hint_pattern in self._compiled_hints:
                if hint_pattern.search(text_to_check):
                    total_matches += 1

        # Normalize to 0.0-1.0, with boost for multiple matches
        base_score = total_matches / max(1, total_possible)
        boosted_score = min(1.0, base_score * 2.0)  # Boost up to 2x

        return boosted_score


@dataclass
class Span:
    """Detected actionable span within a document passage."""

    type: Literal[
        "step",
        "jira",
        "itsm",
        "owner",
        "endpoint",
        "table",
        "form",
        "runbook",
        "pipeline",
        "repo",
        "chat",
        "dashboard",
        "teams",
        "outlook",
    ]
    suite: str  # e.g., "jira", "servicenow", "github", "teams", "outlook"
    capability: str  # e.g., "ticket.create", "ticket.view", "api.try", "contact.open"
    text: str  # matched text
    url: str  # best URL (passage URL or extracted)
    doc_id: str  # source document ID
    offset: int  # character offset in passage
    confidence: float  # suite_confidence * extractor_weight
    attrs: Dict[str, str] = field(default_factory=dict)  # captured attributes


# Capability type mappings
CAPABILITY_TO_TYPE = {
    "ticket.create": "jira",
    "ticket.view": "jira",
    "request.create": "itsm",
    "request.view": "itsm",
    "api.try": "endpoint",
    "form.submit": "form",
    "runbook.run": "runbook",
    "pipeline.trigger": "pipeline",
    "repo.create": "repo",
    "contact.open": "chat",
    "dashboard.view": "dashboard",
    "channel.open": "teams",
    "meeting.schedule": "teams",
    "email.send": "outlook",
    "calendar.create": "outlook",
}

# Capability weights for actionability scoring
CAPABILITY_WEIGHTS = {
    "ticket.create": 1.0,
    "ticket.view": 0.6,
    "request.create": 1.0,
    "request.view": 0.6,
    "api.try": 0.9,
    "form.submit": 0.9,
    "runbook.run": 1.0,
    "pipeline.trigger": 1.0,
    "repo.create": 0.8,
    "contact.open": 0.5,
    "dashboard.view": 0.7,
    "channel.open": 0.5,
    "meeting.schedule": 0.6,
    "email.send": 0.4,
    "calendar.create": 0.4,
    # Generic types have lower weight
    "step": 0.4,
    "owner": 0.3,
    "table": 0.2,
}
