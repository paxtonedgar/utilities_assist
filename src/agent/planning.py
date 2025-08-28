"""Planning helpers for the Orchestrator.

Provides thin, testable functions so the main Orchestrator can remain small.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanningMode(str, Enum):
    SIMPLE = "simple"
    STRUCTURED = "structured"


@dataclass
class Plan:
    steps: List[Dict[str, Any]]
    confidence: float = 0.8
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    expected_answer_shape: str = "definition"
    reasoning: str = ""
    planning_mode: PlanningMode = PlanningMode.SIMPLE


def enhanced_simple_plan(query: str) -> Plan:
    """Rule-based plan used when LLM planning is disabled or times out."""
    steps = [{"type": "search", "query": query, "filters": {}, "k": 15}]
    return Plan(steps=steps, planning_mode=PlanningMode.SIMPLE)


def structured_llm_plan(chat_client: Any, query: str, context: Optional[List[Dict]] = None, timeout_s: float = 0.8) -> Plan:
    """LLM-backed planning (very thin wrapper). Returns a minimal plan object.

    Note: Keep orchestration fast; callers should enforce their own timeouts.
    """
    # Minimal prompt; real prompt building should live elsewhere if needed
    messages = [
        {"role": "user", "content": f"Plan steps to answer: {query}. Return JSON steps only."}
    ]
    try:
        resp = chat_client.chat.completions.create(
            model=getattr(chat_client, "model", None) or "gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        # Extremely defensive parse; if anything goes wrong fall back to simple
        text = resp.choices[0].message.content if resp and resp.choices else ""
        # Best-effort JSON find
        import json, re

        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text or "")
        steps = []
        if match:
            obj = json.loads(match.group(0))
            steps = obj if isinstance(obj, list) else obj.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        if not steps:
            steps = [{"type": "search", "query": query, "filters": {}, "k": 15}]
        return Plan(steps=steps, planning_mode=PlanningMode.STRUCTURED, confidence=0.85)
    except Exception:
        return enhanced_simple_plan(query)

