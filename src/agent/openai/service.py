"""OpenAI service facade for planning and composing.

Nodes call these thin functions; the implementation can swap between
OpenAI Responses API and local fallbacks without changing node code.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.agent.openai.schemas import Plan, Card, Citation, Step, ApiItem

logger = logging.getLogger(__name__)


def _detect_utility_from_query(query: str) -> Optional[str]:
    """Very small heuristic utility detector as a fallback (keeps DRY/YAGNI)."""
    from src.agent.acronym_map import UTILITY_ACRONYMS
    tokens = query.split()
    for t in tokens:
        full = UTILITY_ACRONYMS.get(t.upper())
        if full:
            return full
    return None


def get_plan(normalized_query: str, session_ctx: Optional[Dict] = None) -> Plan:
    """Return a Plan for the turn.

    Implementation note: We keep this local and deterministic for now.
    Later, this can call OpenAI Responses API with JSON schema.
    """
    util = _detect_utility_from_query(normalized_query)
    filters: Dict[str, str] = {}
    if util:
        filters["utility_name"] = util

    # Helpful over concise: include more aspects by default
    aspects = ["overview", "steps", "api"]
    if any(w in normalized_query.lower() for w in ["error", "issue", "debug", "fix"]):
        aspects.append("troubleshoot")

    plan = Plan(aspects=aspects, filters=filters)
    logger.info(
        f"Planner: aspects={plan.aspects} k={plan.k_per_aspect} filters={plan.filters} budgets={plan.budgets}"
    )
    return plan


def compose_card(
    sections: Dict[str, List[Dict]],
    budgets: Dict[str, int],
    utility: Optional[str] = None,
) -> Card:
    """Compose a structured card from small, targeted passages per aspect.

    Implementation: deterministic local composer with strict budgets.
    Later replace with OpenAI Responses API returning JSON.
    """
    def trim(text: str, n: int) -> str:
        t = " ".join((text or "").split())
        if len(t) <= n:
            return t
        # try to end at a sentence boundary
        cut = t[:n]
        if ". " in cut:
            return cut.rsplit(". ", 1)[0] + "."
        return cut + "…"

    def mk_citation(p: Dict) -> Citation:
        title = p.get("title") or "Untitled"
        url = p.get("url") or ""
        return Citation(title=title, url=url)

    card = Card(utility=utility)

    # Overview
    if secs := sections.get("overview"):
        text = trim(secs[0].get("snippet", secs[0].get("text", "")), budgets.get("overview_chars", 500))
        card.overview = {"text": text, "citations": [mk_citation(secs[0])]} 
    else:
        card.unknown_fields.append("overview")

    # Steps
    steps_src = sections.get("steps", [])
    if steps_src:
        cap = budgets.get("steps_chars", 900)
        n = 1
        for p in steps_src:
            txt = trim(p.get("snippet", p.get("text", "")), min(180, cap // 5))
            if txt:
                card.onboarding_steps.append(Step(n=n, text=txt, citation=mk_citation(p)))
                n += 1
        if not card.onboarding_steps:
            card.unknown_fields.append("onboarding_steps")
    else:
        card.unknown_fields.append("onboarding_steps")

    # APIs
    api_src = sections.get("api", [])
    if api_src:
        cap = budgets.get("api_chars", 500)
        for p in api_src:
            name = p.get("title") or p.get("api_name") or "API"
            card.apis.append(ApiItem(name=name, url=p.get("url") or "", citation=mk_citation(p)))
            if len(card.apis) >= 5:
                break
    else:
        card.unknown_fields.append("apis")

    # Troubleshooting (optional text list)
    tr_src = sections.get("troubleshoot", [])
    if tr_src:
        for p in tr_src[:2]:
            url = p.get("url") or ""
            card.links.append({"label": "Troubleshooting", "url": url, "citation": mk_citation(p).__dict__})

    return card

