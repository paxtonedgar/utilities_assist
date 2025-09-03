"""Plan node: replaces fixed intents with dynamic aspects + filters.

Linear graph: summarize -> plan -> search -> compose -> answer
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from .base_node import BaseNodeHandler
from src.agent.services.planner_composer import get_plan

logger = logging.getLogger(__name__)


class PlanNode(BaseNodeHandler):
    def __init__(self):
        super().__init__("plan")

    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        query = state.get("normalized_query") or state.get("original_query", "")
        if not query:
            logger.warning("PlanNode: empty query")
            return {**state, "plan": None}

        session_ctx = {
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
        }
        plan = get_plan(query, session_ctx)

        logger.info(
            f"PlanNode: aspects={plan.aspects}, filters={plan.filters}, k={plan.k_per_aspect}"
        )

        return {
            **state,
            "plan": {
                "aspects": plan.aspects,
                "filters": plan.filters,
                "k_per_aspect": plan.k_per_aspect,
                "budgets": plan.budgets,
            },
            "workflow_path": state.get("workflow_path", []) + ["plan"],
        }
