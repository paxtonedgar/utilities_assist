"""
Step extraction: regex, spaCy, and LLM-based approaches.

Initial implementation provides a pragmatic regex-based extractor that:
- Detects numbered/bulleted lists
- Heuristically extracts imperative phrases at sentence starts
"""

from typing import List, Dict, Any
import re
import hashlib


_BULLET_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|\(\d+\)|[-–•*])\s+(?P<text>.+?)\s*$")
# Swagger/API style endpoint lines, e.g., "GET /v1/apps/{id}" or "POST /api/foo"
_HTTP_PATH_RE = re.compile(r"(?mi)^\s*(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+(/[\w\-./{}:]+)\b.*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize_step_label(text: str) -> str:
    t = " ".join(text.strip().split())
    # Strip trailing punctuation
    return re.sub(r"[.;:]+$", "", t)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def regex_step_extractor(text: str) -> List[Dict[str, Any]]:
    """Return candidate steps from raw text using regex heuristics.

    Output schema (per step):
    {
        "label": str,
        "verb": str | None,
        "obj": str | None,
        "order": int | None,
        "source": "regex",
        "evidence": {"text_snippet": str}
    }
    """
    steps: List[Dict[str, Any]] = []

    # 1) Bulleted/numbered lists (preserve visual order)
    order = 1
    for m in _BULLET_RE.finditer(text):
        snippet = m.group("text").strip()
        label = _normalize_step_label(snippet)
        steps.append(
            {
                "label": label,
                "verb": None,
                "obj": None,
                "order": order,
                "source": "regex",
                "evidence": {"text_snippet": snippet, "snippet_hash": _sha256(snippet)},
            }
        )
        order += 1

    # 2) Swagger-like HTTP endpoint lines (fallback when there are no bullets)
    #    This helps indices like khub-opensearch-swagger-index yield actionable steps.
    if not steps:
        for m in _HTTP_PATH_RE.finditer(text):
            method, path = m.group(1).upper(), m.group(2)
            snippet = f"{method} {path}"
            label = _normalize_step_label(snippet)
            steps.append(
                {
                    "label": label,
                    "verb": method,
                    "obj": path,
                    "order": order,
                    "source": "regex",
                    "evidence": {"text_snippet": snippet, "snippet_hash": _sha256(snippet)},
                }
            )
            order += 1

    return steps


def spacy_step_extractor(*_args, **_kwargs):
    return []


def llm_step_extractor(*_args, **_kwargs):
    return []
