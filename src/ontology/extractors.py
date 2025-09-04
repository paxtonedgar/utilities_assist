"""
Step extraction: regex, spaCy, and LLM-based approaches.

Initial implementation provides a pragmatic regex-based extractor that:
- Detects numbered/bulleted lists
- Heuristically extracts imperative phrases at sentence starts
"""

from typing import List, Dict, Any, Optional
import re


_BULLET_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|\(\d+\)|[-–•*])\s+(?P<text>.+?)\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_IMPERATIVE_VERBS = (
    "configure|install|deploy|verify|submit|enable|provision|request|register|create|update|run|build|test|restart|start|stop|set|add|remove|grant|apply|open|click|upload|download|compile|execute|attach|authorize|approve|rollback"
)
_IMPERATIVE_RE = re.compile(rf"^(?P<verb>{_IMPERATIVE_VERBS})\b[\w\W]*", re.IGNORECASE)


def _normalize_step_label(text: str) -> str:
    t = " ".join(text.strip().split())
    # Strip trailing punctuation
    return re.sub(r"[.;:]+$", "", t)


def _extract_verb_obj(text: str) -> (Optional[str], Optional[str]):
    m = _IMPERATIVE_RE.match(text.strip())
    if not m:
        return None, None
    verb = m.group("verb").lower()
    # crude object heuristic: text after verb
    obj = text.strip()[len(m.group("verb")) :].strip(" :.-") or None
    return verb, obj


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

    # 1) Bulleted/numbered lists (preserve order)
    order = 1
    for m in _BULLET_RE.finditer(text):
        snippet = m.group("text").strip()
        label = _normalize_step_label(snippet)
        verb, obj = _extract_verb_obj(label)
        steps.append(
            {
                "label": label,
                "verb": verb,
                "obj": obj,
                "order": order,
                "source": "regex",
                "evidence": {"text_snippet": snippet},
            }
        )
        order += 1

    # 2) Imperative sentences at paragraph starts (fallback)
    if not steps:
        sentences = _SENTENCE_SPLIT.split(text.strip()) if text.strip() else []
        local_order = 1
        for sent in sentences[:10]:  # limit for performance
            if _IMPERATIVE_RE.match(sent.strip()):
                label = _normalize_step_label(sent)
                verb, obj = _extract_verb_obj(label)
                steps.append(
                    {
                        "label": label,
                        "verb": verb,
                        "obj": obj,
                        "order": local_order,
                        "source": "regex",
                        "evidence": {"text_snippet": sent.strip()},
                    }
                )
                local_order += 1

    return steps


def spacy_step_extractor(text: str, nlp: Optional[object] = None) -> List[Dict[str, Any]]:
    """Return candidate steps using dependency parsing for imperatives.

    Placeholder: returns empty list unless wired with spaCy pipeline.
    """
    return []


def llm_step_extractor(passages: List[str], llm: Optional[object] = None) -> List[Dict[str, Any]]:
    """Return steps by prompting an LLM to extract an ordered list in JSON.

    Placeholder: returns empty list to keep the pipeline deterministic in tests.
    """
    return []
