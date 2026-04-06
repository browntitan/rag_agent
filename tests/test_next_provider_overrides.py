from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.providers import ProviderBundle
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.providers import factory as provider_factory
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop, QueryLoopResult


class RecordingChatModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": list(messages), "config": dict(config or {})})
        return SimpleNamespace(content=self.response_text)


class FakeRagContract:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [],
            "used_citation_ids": [],
            "warnings": [],
            "followups": [],
        }


def _provider_bundle(*, chat_text: str, judge_text: str = "judge") -> ProviderBundle:
    return ProviderBundle(
        chat=RecordingChatModel(chat_text),
        judge=RecordingChatModel(judge_text),
        embeddings=object(),
    )


def _runtime_settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        skills_dir=repo_root / "data" / "skills",
        agents_dir=repo_root / "data" / "agents",
        llm_provider="ollama",
        judge_provider="ollama",
        ollama_chat_model="base-chat",
        ollama_judge_model="base-judge",
        runtime_events_enabled=False,
        max_worker_concurrency=2,
        planner_max_tasks=4,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        enable_coordinator_mode=False,
        agent_chat_model_overrides={},
        agent_judge_model_overrides={},
    )


def test_agent_provider_resolver_returns_base_bundle_without_override(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")

    resolver = provider_factory.AgentProviderResolver(settings, base_providers)

    assert resolver.for_agent("general") is base_providers


def test_agent_provider_resolver_reuses_cached_bundle_for_identical_override_tuples(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    settings.agent_chat_model_overrides = {
        "general": "gpt-oss:20b",
        "utility": "gpt-oss:20b",
    }
    settings.agent_judge_model_overrides = {
        "general": "gpt-oss:20b",
        "utility": "gpt-oss:20b",
    }
    base_providers = _provider_bundle(chat_text="base")
    calls: list[tuple[str | None, str | None, object]] = []

    def fake_build_providers(settings_arg, *, embeddings=None, chat_model_override=None, judge_model_override=None):
        del settings_arg
        calls.append((chat_model_override, judge_model_override, embeddings))
        return ProviderBundle(chat=object(), judge=object(), embeddings=embeddings)

    monkeypatch.setattr(provider_factory, "build_providers", fake_build_providers)

    resolver = provider_factory.AgentProviderResolver(settings, base_providers)
    general_bundle = resolver.for_agent("general")
    utility_bundle = resolver.for_agent("utility")

    assert general_bundle is utility_bundle
    assert calls == [("gpt-oss:20b", "gpt-oss:20b", base_providers.embeddings)]


def test_runtime_kernel_passes_agent_specific_providers_into_tool_context_and_query_loop(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    override_providers = _provider_bundle(chat_text="override")
    kernel = RuntimeKernel(settings, providers=base_providers, stores=SimpleNamespace())
    session_state = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = kernel.registry.get("general")
    assert agent is not None
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        kernel,
        "resolve_providers_for_agent",
        lambda agent_name: override_providers if agent_name == "general" else base_providers,
    )
    monkeypatch.setattr(
        kernel,
        "_build_tools",
        lambda agent_arg, tool_context: captured.setdefault("tool_context_providers", tool_context.providers) or [],
    )

    def fake_run(agent_arg, session_state_arg, *, user_text, providers=None, tool_context=None, tools=None, task_payload=None):
        del agent_arg, user_text, tools, task_payload
        captured["loop_providers"] = providers
        captured["loop_tool_context_providers"] = getattr(tool_context, "providers", None)
        return QueryLoopResult(text="override result", messages=list(session_state_arg.messages), metadata={})

    monkeypatch.setattr(kernel.query_loop, "run", fake_run)

    result = kernel.run_agent(agent, session_state, user_text="hello", callbacks=[])

    assert result.text == "override result"
    assert captured["tool_context_providers"] is override_providers
    assert captured["loop_providers"] is override_providers
    assert captured["loop_tool_context_providers"] is override_providers


def test_query_loop_uses_override_providers_for_planner_and_finalizer_and_verifier(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text='{"summary":"base","tasks":[]}')
    override_providers = _provider_bundle(
        chat_text='{"status":"revise","summary":"override verifier","issues":["needs work"],"feedback":"fix it"}'
    )
    loop = QueryLoop(settings=settings, providers=base_providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    planner = AgentDefinition(name="planner", mode="planner", prompt_file="planner_agent.md")
    finalizer = AgentDefinition(name="finalizer", mode="finalizer", prompt_file="finalizer_agent.md")
    verifier = AgentDefinition(name="verifier", mode="verifier", prompt_file="verifier_agent.md")

    planner_override = _provider_bundle(chat_text='{"summary":"override planner","tasks":[]}')
    planner_result = loop.run(planner, session, user_text="plan it", providers=planner_override)
    assert json.loads(planner_result.text)["summary"] == "override planner"

    finalizer_override = _provider_bundle(chat_text="override finalizer")
    finalizer_result = loop.run(
        finalizer,
        session,
        user_text="finalize it",
        providers=finalizer_override,
        task_payload={"partial_answer": "fallback"},
    )
    assert finalizer_result.text == "override finalizer"

    verifier_result = loop.run(
        verifier,
        session,
        user_text="verify it",
        providers=override_providers,
        task_payload={"partial_answer": "candidate"},
    )
    verification = json.loads(verifier_result.text)
    assert verification["summary"] == "override verifier"
    assert verification["status"] == "revise"


def test_query_loop_uses_override_providers_for_rag(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    override_providers = _provider_bundle(chat_text="override", judge_text="override judge")
    loop = QueryLoop(settings=settings, providers=base_providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(settings_arg, stores_arg, *, providers, session, query, conversation_context, preferred_doc_ids, must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks, skill_context, task_context):
        del settings_arg, stores_arg, session, query, conversation_context, preferred_doc_ids
        del must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks, skill_context, task_context
        captured["providers"] = providers
        return FakeRagContract("override rag answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(agent, session, user_text="cite docs", providers=override_providers)

    assert captured["providers"] is override_providers
    assert "override rag answer" in result.text


def test_job_runner_resolves_providers_for_worker_agent_name(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    utility_providers = _provider_bundle(chat_text="utility override")
    kernel = RuntimeKernel(settings, providers=base_providers, stores=SimpleNamespace())
    seen: list[str] = []

    def fake_resolve(agent_name: str):
        seen.append(agent_name)
        if agent_name == "utility":
            return utility_providers
        return base_providers

    monkeypatch.setattr(kernel, "resolve_providers_for_agent", fake_resolve)
    monkeypatch.setattr(
        kernel.query_loop,
        "run",
        lambda agent, session_state, *, user_text, providers=None, tool_context=None, tools=None, task_payload=None: QueryLoopResult(
            text=f"worker:{agent.name}",
            messages=list(session_state.messages),
            metadata={},
        ),
    )

    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="worker prompt",
        session_id="tenant:user:conv",
        description="worker",
        metadata={
            "session_state": {
                "tenant_id": "tenant",
                "user_id": "user",
                "conversation_id": "conv",
            },
            "worker_request": {
                "task_id": "task_1",
                "skill_queries": [],
            },
        },
    )

    result = kernel._job_runner(job)

    assert result == "worker:utility"
    assert seen == ["utility"]
