from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot_next.router.llm_router import route_turn
from agentic_chatbot_next.router.policy import choose_agent_name


def test_route_turn_suggests_data_analyst_for_csv_queries() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Analyze this CSV and compute total revenue by region.",
        has_attachments=False,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"
    assert choose_agent_name(settings, decision) == "data_analyst"


def test_route_turn_suggests_coordinator_for_multistep_comparison() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Compare the uploaded contracts, verify the differences, then synthesize a recommendation.",
        has_attachments=True,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "coordinator"
    assert choose_agent_name(settings, decision) == "coordinator"
