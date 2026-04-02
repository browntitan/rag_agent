from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.session import ChatSession
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop


def _settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=repo_root / "data" / "agents",
        llm_provider="ollama",
        judge_provider="ollama",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
    )


def test_stub_kernel_persists_turn_state_transcript_and_events(tmp_path: Path) -> None:
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
    )
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["hello from next runtime"]),
        judge=FakeListChatModel(responses=["unused"]),
    )
    kernel = RuntimeKernel(_settings(tmp_path), providers=providers, stores=None)
    result = kernel.process_turn(session, user_text="hello foundation")

    assert result == "hello from next runtime"
    paths = RuntimePaths.from_settings(_settings(tmp_path))
    session_dir = paths.session_dir(session.session_id)
    state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    transcript_rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert state["active_agent"] == "general"
    assert state["workspace_root"] == str(paths.workspace_dir(session.session_id))
    assert [row["message"]["role"] for row in transcript_rows] == ["user", "assistant"]
    assert {row["event_type"] for row in event_rows} >= {
        "turn_accepted",
        "agent_run_started",
        "agent_run_completed",
        "turn_completed",
    }
    assert [message.type for message in session.messages] == ["system", "human", "ai"]


class ExplodingQueryLoop(QueryLoop):
    def run(self, agent, session_state, *, user_text: str, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"boom:{agent.name}:{user_text}")


def test_kernel_persists_user_turn_before_executor_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = AgentRegistry(settings.agents_dir)
    kernel = RuntimeKernel(settings, registry=registry, query_loop=ExplodingQueryLoop())
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="failure-case",
    )

    with pytest.raises(RuntimeError, match="boom:general:fail now"):
        kernel.process_turn(session, user_text="fail now")

    paths = RuntimePaths.from_settings(settings)
    session_dir = paths.session_dir(session.session_id)
    state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    transcript_rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(state["messages"]) == 1
    assert transcript_rows[0]["message"]["content"] == "fail now"
    assert "turn_failed" in {row["event_type"] for row in event_rows}


def test_kernel_serializes_parallel_worker_batches_for_local_ollama(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[{"mode": "parallel"}, {"mode": "parallel"}],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is False


def test_kernel_keeps_parallel_worker_batches_for_non_ollama_providers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.llm_provider = "azure"
    settings.judge_provider = "azure"
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[{"mode": "parallel"}, {"mode": "parallel"}],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is True


def test_kernel_serializes_parallel_worker_batches_when_provider_objects_are_ollama(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.llm_provider = ""
    settings.judge_provider = ""

    class FakeOllamaChat:
        pass

    FakeOllamaChat.__module__ = "langchain_ollama.chat_models"

    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeOllamaChat(),
            judge=FakeOllamaChat(),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[{"mode": "parallel"}, {"mode": "parallel"}],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is False
