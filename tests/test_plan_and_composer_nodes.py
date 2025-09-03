import asyncio

from src.agent.nodes.plan import PlanNode
from src.agent.nodes.composer_node import ComposerNode


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_plan_node_produces_plan_dict():
    node = PlanNode()
    state = {"normalized_query": "tell me about CIU", "workflow_path": []}
    out = run(node.execute(state, config={}))

    assert "plan" in out and isinstance(out["plan"], dict)
    assert out["plan"]["aspects"], "plan.aspects should not be empty"
    assert "filters" in out["plan"]


def test_composer_node_renders_markdown_from_sections():
    node = ComposerNode()
    state = {
        "plan": {
            "budgets": {"overview_chars": 500, "steps_chars": 900, "api_chars": 500},
            "filters": {"utility_name": "Customer Interaction Utility"},
        },
        "search_sections": {
            "overview": [
                {
                    "doc_id": "d1",
                    "title": "Customer Interaction Utility",
                    "url": "https://example/ciu",
                    "snippet": "CIU overview text",
                    "score": 0.9,
                }
            ],
            "steps": [
                {
                    "doc_id": "d2",
                    "title": "Onboarding",
                    "url": "https://example/ciu/onboarding",
                    "snippet": "- Step one\n- Step two",
                    "score": 0.7,
                }
            ],
            "api": [
                {
                    "doc_id": "d3",
                    "title": "CIU Tickets API",
                    "url": "https://example/ciu/api",
                    "snippet": "GET /tickets",
                    "score": 0.6,
                }
            ],
        },
        "workflow_path": [],
    }

    out = run(node.execute(state, config={}))
    assert out.get("final_briefing"), "composer should produce markdown"
    assert "Overview" in out["final_briefing"]
    assert "Steps & Procedures" in out["final_briefing"]
    assert "API Reference" in out["final_briefing"]

