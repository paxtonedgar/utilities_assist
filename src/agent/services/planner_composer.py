"""OpenAI service facade for planning and composing.

Nodes call these thin functions; the implementation can swap between
OpenAI Responses API and local fallbacks without changing node code.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.services.models import Plan, Card, Citation, Step, ApiItem
from src.infra.settings import get_settings
from src.infra.resource_manager import get_resources

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
    settings = get_settings()
    util = _detect_utility_from_query(normalized_query)
    filters: Dict[str, str] = {}
    if util:
        filters["utility_name"] = util

    # Helpful over concise: include more aspects by default
    aspects = ["overview", "steps", "api"]
    if any(w in normalized_query.lower() for w in ["error", "issue", "debug", "fix"]):
        aspects.append("troubleshoot")

    # Try OpenAI planner if enabled; fall back to local
    if settings.enable_openai_planner:
        try:
            # Reuse the chat client created at startup (resource manager singleton)
            resources = get_resources()
            client = resources.chat_client if resources else None
            if client is None:
                raise RuntimeError("chat_client not initialized in resources")
            system = (
                "You are a planner for an enterprise utilities assistant. "
                "Return ONLY JSON. Schema: {aspects: string[], filters: object, k_per_aspect: number, budgets: object}."
            )
            schema = {
                "type": "object",
                "properties": {
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["overview", "steps", "api", "troubleshoot"],
                        },
                    },
                    "filters": {"type": "object"},
                    "k_per_aspect": {"type": "integer"},
                    "budgets": {"type": "object"},
                },
                "required": ["aspects", "filters", "k_per_aspect", "budgets"],
                "additionalProperties": False,
            }
            # Use json_object for broad Azure compatibility
            resp = client.chat.completions.create(
                model=(settings.openai_planner_model or settings.chat.model),
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Query: {normalized_query}"},
                ],
                timeout=settings.openai_timeout_ms / 1000.0,
            )
            content = resp.choices[0].message.content if resp.choices else "{}"
            import json

            data = json.loads(content or "{}")
            aspects = data.get("aspects") or ["overview", "steps", "api"]
            filters = data.get("filters") or {}
            if util and "utility_name" not in filters:
                filters["utility_name"] = util
            kpa = int(data.get("k_per_aspect", 3))
            budgets = data.get("budgets") or Plan().budgets
            strategies = data.get("aspect_strategies") or Plan().aspect_strategies
            return Plan(
                aspects=aspects,
                filters=filters,
                k_per_aspect=kpa,
                budgets=budgets,
                aspect_strategies=strategies,
            )
        except Exception as e:  # Fallback to local
            logger.warning(f"OpenAI planner failed, using local: {e}")

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

    settings = get_settings()

    # Attempt OpenAI composer if enabled
    if settings.enable_openai_composer:
        try:
            resources = get_resources()
            client = resources.chat_client if resources else None
            if client is None:
                raise RuntimeError("chat_client not initialized in resources")
            system = (
                "Compose a structured knowledge card. Return ONLY JSON with fields: "
                "utility, overview{text,citations[{title,url}]}, onboarding_steps[{n,text,citation}], "
                "apis[{name,url,citation}], environments[{name,url,auth?,citation}], links[{label,url,citation}], "
                "unknown_fields. Respect char budgets and always include citations."
            )
            schema = {
                "type": "object",
                "properties": {
                    "utility": {"type": ["string", "null"]},
                    "overview": {"type": ["object", "null"]},
                    "onboarding_steps": {"type": "array"},
                    "apis": {"type": "array"},
                    "environments": {"type": "array"},
                    "links": {"type": "array"},
                    "unknown_fields": {"type": "array"},
                },
                "required": [
                    "utility",
                    "overview",
                    "onboarding_steps",
                    "apis",
                    "environments",
                    "links",
                    "unknown_fields",
                ],
                "additionalProperties": False,
            }
            import json

            user_payload = json.dumps(
                {"utility": utility, "budgets": budgets, "sections": sections}
            )
            resp = client.chat.completions.create(
                model=(settings.openai_composer_model or settings.chat.model),
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_payload},
                ],
                timeout=settings.openai_timeout_ms / 1000.0,
            )
            content = resp.choices[0].message.content if resp.choices else "{}"
            data = json.loads(content or "{}")

            # Map into Card dataclass minimally (use local renderer later)
            card = Card(
                utility=data.get("utility"),
                overview=data.get("overview"),
                onboarding_steps=[
                    Step(
                        n=item.get("n", i + 1),
                        text=item.get("text", ""),
                        citation=Citation(**item.get("citation", {"title": "", "url": ""})),
                    )
                    for i, item in enumerate(data.get("onboarding_steps", [])[:9])
                ],
                apis=[
                    ApiItem(
                        name=item.get("name", "API"),
                        url=item.get("url", ""),
                        citation=Citation(**item.get("citation", {"title": "", "url": ""})),
                    )
                    for item in data.get("apis", [])[:5]
                ],
                environments=data.get("environments", []),
                links=data.get("links", []),
                unknown_fields=data.get("unknown_fields", []),
            )
            return card
        except Exception as e:
            logger.warning(f"OpenAI composer failed, using local: {e}")

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
