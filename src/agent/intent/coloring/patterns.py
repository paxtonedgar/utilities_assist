"""Shared regex patterns for coloring system - compiled once for performance."""

import re
from typing import Dict, Pattern

# Compile patterns once for performance and thread-safety
SHARED_PATTERNS: Dict[str, Pattern] = {
    # Actionability patterns
    "http_endpoint": re.compile(
        r"\b(?:POST|GET|PUT|DELETE|PATCH)\s+/[\w/]+", re.IGNORECASE
    ),
    "api_path": re.compile(r"/[\w/]+(?:\?[\w=&]+)?"),
    "ticket_verbs": re.compile(
        r"\b(?:create|submit|open|file)\s+(?:a\s+)?(?:ticket|request|issue)",
        re.IGNORECASE,
    ),
    "step_indicators": re.compile(
        r"\b(?:step\s*\d+|steps?|procedure|runbook|guide)", re.IGNORECASE
    ),
    "form_patterns": re.compile(
        r"\b(?:form|submit|fill\s+out|pipeline)", re.IGNORECASE
    ),
    "channel_patterns": re.compile(
        r"\b(?:channel|join|teams|slack|chat)", re.IGNORECASE
    ),
    # Specificity anchors (distinct patterns to avoid double-counting)
    "project_key": re.compile(r"\b[A-Z]{2,5}-\d+\b"),  # ABC-123
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),  # v1.2.3, 2.1
    "env_tag": re.compile(r"\b(?:dev|test|staging|prod|production)\b", re.IGNORECASE),
    "uuid": re.compile(
        r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.IGNORECASE
    ),
    "id_pattern": re.compile(r"\b(?:id|ID)[\s=:]+\w+"),
    # Troubleshooting indicators
    "error_codes": re.compile(
        r"\b(?:[45]\d{2}|error|exception|failed?|timeout)", re.IGNORECASE
    ),
    "stack_trace": re.compile(
        r"\b(?:stack\s*trace|traceback|exception\s*in)", re.IGNORECASE
    ),
    # Time urgency patterns
    "urgency_soft": re.compile(r"\b(?:soon|today|asap|urgent|priority)", re.IGNORECASE),
    "urgency_hard": re.compile(
        r"\b(?:now|immediately|critical|emergency|sla|deadline)", re.IGNORECASE
    ),
    # Safety patterns (PII/credentials)
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "token_pattern": re.compile(
        r"\b(?:token|key|secret|password)[\s=:]+\S{16,}", re.IGNORECASE
    ),
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credential_keywords": re.compile(
        r"\b(?:password|token|key|secret)[\s=:]+", re.IGNORECASE
    ),
}


def count_distinct_matches(text: str, pattern_names: list) -> int:
    """Count distinct matches across multiple patterns to avoid double-counting."""
    all_matches = set()

    for pattern_name in pattern_names:
        if pattern_name in SHARED_PATTERNS:
            matches = SHARED_PATTERNS[pattern_name].findall(text)
            all_matches.update(matches)

    return len(all_matches)


def has_pattern_match(text: str, pattern_name: str) -> bool:
    """Check if text matches a specific pattern."""
    if pattern_name not in SHARED_PATTERNS:
        return False
    return bool(SHARED_PATTERNS[pattern_name].search(text))


def extract_pattern_matches(text: str, pattern_name: str) -> list:
    """Extract all matches for a specific pattern."""
    if pattern_name not in SHARED_PATTERNS:
        return []
    return SHARED_PATTERNS[pattern_name].findall(text)
