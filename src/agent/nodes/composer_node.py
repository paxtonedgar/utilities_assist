"""Composer node: builds a structured card from per-aspect passages.

Keeps output as JSON-like dict; Answer node/presenter can render.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from .base_node import BaseNodeHandler
from src.agent.services.planner_composer import compose_card
from src.infra.resource_manager import get_resources
from src.agent.constants import ORIGINAL_QUERY, SEARCH_QUERY, NORMALIZED_QUERY

logger = logging.getLogger(__name__)


class ComposerNode(BaseNodeHandler):
    def __init__(self):
        super().__init__("compose")

    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        budgets = plan.get("budgets", {})
        utility = (plan.get("filters") or {}).get("utility_name")

        # Expect grouped results in state.search_sections: {aspect: [passage_dict]}
        sections: Dict[str, List[Dict]] = state.get("search_sections", {})
        if not sections:
            logger.info("ComposerNode: no sections to compose; returning minimal message")
            return {
                **state,
                "final_briefing": "No information available to compose.",
                "workflow_path": state.get("workflow_path", []) + ["compose_empty"],
            }

        # Optional direct LLM answering without trimming
        resources = get_resources()
        settings = resources.settings if resources else None
        if settings and getattr(settings, "enable_llm_direct_answer", False) and resources and resources.chat_client:
            question = state.get(SEARCH_QUERY) or state.get(ORIGINAL_QUERY) or state.get(NORMALIZED_QUERY) or ""
            logger.info("ComposerNode: direct LLM answer enabled (no trimming)")
            ctx_lines: List[str] = []
            for aspect, items in sections.items():
                ctx_lines.append(f"## {aspect}")
                for i, p in enumerate(items, 1):
                    title = p.get("title") or "Untitled"
                    url = p.get("url") or ""
                    text = p.get("snippet") or ""
                    ctx_lines.append(f"### {i}. {title}")
                    if url:
                        ctx_lines.append(f"Source: {url}")
                    ctx_lines.append(text)
                    ctx_lines.append("")
            context_blob = "\n".join(ctx_lines)
            system = (
                "You are an enterprise utilities assistant. Answer using ONLY the provided context. "
                "Cite sources inline with Markdown links where possible. Be precise and comprehensive."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_blob}"},
            ]
            try:
                resp = resources.chat_client.chat.completions.create(
                    model=settings.chat.model,
                    temperature=0.2,
                    messages=messages,
                    timeout=settings.openai_timeout_ms / 1000.0,
                )
                answer = resp.choices[0].message.content if resp and resp.choices else ""
                answer = answer or "No answer generated."
                return {
                    **state,
                    "final_briefing": answer,
                    "workflow_path": state.get("workflow_path", []) + ["compose_direct_llm"],
                }
            except Exception as e:
                logger.error(f"Direct LLM answering failed: {e}")

        card = compose_card(sections=sections, budgets=budgets, utility=utility)

        # Minimal Markdown rendering (UI can do richer rendering)
        lines = []
        if card.utility:
            lines.append(f"# {card.utility}")
        if card.overview and card.overview.get("text"):
            lines.append("\n## Overview\n")
            lines.append(card.overview["text"])
        if card.onboarding_steps:
            lines.append("\n## Steps & Procedures\n")
            for s in card.onboarding_steps[:9]:
                lines.append(f"{s.n}. {s.text}")
        if card.apis:
            lines.append("\n## API Reference\n")
            for a in card.apis[:5]:
                label = a.name or "API"
                if a.url:
                    lines.append(f"- [{label}]({a.url})")
                else:
                    lines.append(f"- {label}")

        final_md = "\n".join(lines).strip() or "No information found."

        return {
            **state,
            "card": {
                "utility": card.utility,
                "unknown_fields": card.unknown_fields,
            },
            "final_briefing": final_md,
            "workflow_path": state.get("workflow_path", []) + ["compose"],
        }
