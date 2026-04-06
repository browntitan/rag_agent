from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.router.router import RouterDecision
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


def test_route_turn_suggests_rag_worker_for_grounded_document_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="What are the key implementation details in the architecture docs? Cite your sources.",
        has_attachments=False,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


def test_choose_agent_name_accepts_registry_defined_top_level_specialist(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "general.md").write_text(
        """---
name: general
mode: react
description: default generalist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text"}
---
general
""",
        encoding="utf-8",
    )
    (agents_dir / "policy_specialist.md").write_text(
        """---
name: policy_specialist
mode: react
description: policy-focused top-level specialist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["rag_agent_tool"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "router_or_delegated", "expected_output": "user_text"}
---
policy specialist
""",
        encoding="utf-8",
    )
    registry = AgentRegistry(agents_dir)
    settings = SimpleNamespace(enable_coordinator_mode=False)
    decision = RouterDecision(
        route="AGENT",
        confidence=0.82,
        reasons=["llm_router"],
        suggested_agent="policy_specialist",
        router_method="llm",
    )

    assert choose_agent_name(settings, decision, registry=registry) == "policy_specialist"


def test_route_turn_allows_llm_router_to_suggest_registry_defined_specialist(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "general.md").write_text(
        """---
name: general
mode: react
description: default generalist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text"}
---
general
""",
        encoding="utf-8",
    )
    (agents_dir / "policy_specialist.md").write_text(
        """---
name: policy_specialist
mode: react
description: policy-focused top-level specialist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["rag_agent_tool"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "router_or_delegated", "expected_output": "user_text"}
---
policy specialist
""",
        encoding="utf-8",
    )
    registry = AgentRegistry(agents_dir)

    class FakeJudge:
        def invoke(self, messages):
            del messages
            return '{"route":"AGENT","confidence":0.92,"reasoning":"policy specialist requested","suggested_agent":"policy_specialist"}'

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_confidence_threshold=0.95,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=FakeJudge()),
        user_text="Review the policy changes.",
        has_attachments=False,
        force_agent=False,
        registry=registry,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "policy_specialist"
    assert choose_agent_name(settings, decision, registry=registry) == "policy_specialist"


def test_route_turn_force_agent_preserves_rag_worker_for_grounded_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=True,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Summarize the architecture docs and cite your sources.",
        has_attachments=False,
        force_agent=True,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"
