import pytest

from src.agent.services.planner_composer import get_plan, compose_card


def test_get_plan_infers_aspects_and_filters_for_ciu():
    query = "tell me about CIU"
    plan = get_plan(query, session_ctx={"user_id": "u", "thread_id": "t"})

    # Plan should propose helpful aspects
    assert isinstance(plan.aspects, list) and plan.aspects, "aspects should not be empty"
    assert any(a in plan.aspects for a in ["overview", "steps", "api"])  # helpful defaults

    # Should detect utility from acronym map when possible
    if plan.filters:
        assert plan.filters.get("utility_name"), "expected utility_name in filters for CIU"

    # Budgets should be generous (helpful over concise)
    assert plan.budgets.get("overview_chars", 0) >= 450
    assert plan.k_per_aspect >= 2


def test_compose_card_minimal_sections():
    sections = {
        "overview": [
            {
                "doc_id": "d1",
                "title": "Customer Interaction Utility",
                "url": "https://example/ciu",
                "snippet": "CIU helps with customer interactions across channels.",
                "score": 0.9,
            }
        ],
        "steps": [
            {
                "doc_id": "d2",
                "title": "Onboarding CIU",
                "url": "https://example/ciu/onboarding",
                "snippet": "- Create dependency in Jira\n- Obtain client IDs\n- Request access",
                "score": 0.7,
            }
        ],
        "api": [
            {
                "doc_id": "d3",
                "title": "CIU Tickets API",
                "url": "https://example/ciu/api/tickets",
                "snippet": "GET /tickets",
                "score": 0.5,
            }
        ],
    }

    budgets = {"overview_chars": 500, "steps_chars": 900, "api_chars": 500}
    card = compose_card(sections=sections, budgets=budgets, utility="Customer Interaction Utility")

    assert card.utility == "Customer Interaction Utility"
    assert card.overview and card.overview.get("text")
    assert card.onboarding_steps, "expected at least one step"
    assert card.apis and len(card.apis) >= 1
