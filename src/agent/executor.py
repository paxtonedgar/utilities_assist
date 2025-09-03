"""Execution helpers for plans.

Keeps execution orchestration minimal and testable.
"""

from typing import Any, Dict, List


async def execute_steps(steps: List[Dict[str, Any]], resources: Any) -> Dict[str, Any]:
    """Trivial executor stub – returns first search-like step for now.

    This avoids deep refactors in Orchestrator while giving us a seam to grow.
    """
    if not steps:
        return {"status": "no_steps"}

    # Future: dispatch by step["type"] (search, verify, present, action, etc.)
    return {"status": "planned", "first_step": steps[0]}

