"""Composer node: builds a structured card from per-aspect passages.

Keeps output as JSON-like dict; Answer node/presenter can render.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from .base_node import BaseNodeHandler
from src.agent.services.planner_composer import compose_card

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
