"""Verification helpers.

Shared, minimal verification utilities to keep nodes thin.
"""

from typing import Any, Dict


def verify_answer_basic(answer: str, context: str, query: str) -> Dict[str, Any]:
    """Lightweight verification facade that defers to node-level checks.

    This exists so the Orchestrator (or tools) can import a stable function
    without duplicating the verification logic.
    """
    from src.agent.nodes.processing_nodes import verify_answer  # reuse implementation

    return verify_answer(answer, context, query)

