from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.session import ChatSession


def _repo_agents_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "agents"


def _make_runtime_settings(tmp_path: Path, *, agents_dir: Path | None = None) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        max_worker_concurrency=2,
        agents_dir=agents_dir or _repo_agents_dir(),
        skills_dir=repo_root / "data" / "skills",
        runtime_events_enabled=True,
        enable_coordinator_mode=False,
        planner_max_tasks=4,
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
    )


def test_transcript_store_round_trips_session_state_and_transcript(tmp_path: Path):
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(_make_runtime_settings(tmp_path)))
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
    )
    session.append_message("user", "Hello runtime")
    session.scratchpad["note"] = "persist me"

    store.persist_session_state(session)
    store.append_session_transcript(session.session_id, {"kind": "message", "content": "Hello runtime"})

    loaded = store.load_session_state(session.session_id)
    transcript = store.load_session_transcript(session.session_id)

    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.messages[0].content == "Hello runtime"
    assert loaded.scratchpad == {"note": "persist me"}
    assert transcript == [{"kind": "message", "content": "Hello runtime"}]


def test_job_manager_resumes_waiting_job_after_reinstantiation(tmp_path: Path):
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(_make_runtime_settings(tmp_path)))
    manager_a = RuntimeJobManager(store)
    job = manager_a.create_job(
        agent_name="utility",
        prompt="original prompt",
        session_id="tenant:user:conv",
        description="resume test",
        metadata={"session_state": {"tenant_id": "tenant", "user_id": "user", "conversation_id": "conv"}},
    )
    job.status = "waiting_message"
    store.persist_job_state(job)

    queued = manager_a.enqueue_message(job.job_id, "continue with this")
    assert queued is not None

    manager_b = RuntimeJobManager(store)

    def runner(record):
        mailbox = manager_b.drain_mailbox(record.job_id)
        text = " | ".join(message.content for message in mailbox)
        return f"{record.prompt} :: {text}"

    resumed = manager_b.continue_job(job.job_id, runner)
    assert resumed is not None

    deadline = time.time() + 5
    completed = None
    while time.time() < deadline:
        completed = manager_b.get_job(job.job_id)
        if completed and completed.status == "completed":
            break
        time.sleep(0.05)

    assert completed is not None
    assert completed.status == "completed"
    assert completed.result_summary == "original prompt :: continue with this"
    assert Path(completed.artifact_dir).is_dir()


def test_agent_registry_applies_file_overrides(tmp_path: Path):
    definitions_dir = tmp_path / "agents"
    definitions_dir.mkdir()
    (definitions_dir / "general.md").write_text(
        """---
name: general
mode: react
description: overridden general agent
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 4
allow_background_jobs: false
metadata: {"role_kind": "top_level"}
---
override
""",
        encoding="utf-8",
    )

    registry = AgentRegistry(definitions_dir)
    general = registry.get("general")

    assert general is not None
    assert general.description == "overridden general agent"
    assert general.allowed_tools == ["calculator"]
    assert general.max_steps == 3


def test_runtime_agent_matrix_matches_alignment_expectations(tmp_path: Path):
    registry = AgentRegistry(_make_runtime_settings(tmp_path).agents_dir)

    general = registry.get("general")
    coordinator = registry.get("coordinator")
    verifier = registry.get("verifier")
    memory_maintainer = registry.get("memory_maintainer")

    assert general is not None
    assert coordinator is not None
    assert verifier is not None
    assert memory_maintainer is not None

    assert general.allowed_worker_agents == ["coordinator", "memory_maintainer"]
    assert coordinator.mode == "coordinator"
    assert "planner" in coordinator.allowed_worker_agents
    assert verifier.prompt_file == "verifier_agent.md"
    assert memory_maintainer.metadata["role_kind"] == "maintenance"


def test_job_runner_persists_updated_worker_session_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="original worker prompt",
        session_id="tenant:user:conv",
        description="worker state persistence",
        metadata={"session_state": {"tenant_id": "tenant", "user_id": "user", "conversation_id": "conv"}},
    )

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        return SimpleNamespace(
            text="worker complete",
            messages=[
                RuntimeMessage(role="user", content=user_text),
                RuntimeMessage(role="assistant", content="worker complete"),
            ],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel._job_runner(job)

    refreshed = kernel.job_manager.get_job(job.job_id)
    assert refreshed is not None
    persisted = refreshed.metadata["session_state"]
    assert [item["content"] for item in persisted["messages"]] == [
        "original worker prompt",
        "worker complete",
    ]


def test_process_agent_turn_syncs_user_message_back_to_chat_session_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )

    def fail_run_agent(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(kernel, "run_agent", fail_run_agent)

    with pytest.raises(RuntimeError, match="boom"):
        kernel.process_agent_turn(session, user_text="persist this turn")

    assert [getattr(message, "content", "") for message in session.messages] == ["persist this turn"]
    stored = kernel.transcript_store.load_session_state(session.session_id)
    assert stored is not None
    assert [message.content for message in stored.messages] == ["persist this turn"]


def test_coordinator_runs_planner_workers_finalizer_and_verifier_with_scoped_worker_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    session_state.append_message("assistant", "prior parent context")
    worker_histories = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Gather evidence",
                                "executor": "rag_worker",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Find the relevant evidence.",
                                "doc_scope": ["contract-a"],
                                "skill_queries": ["citation hygiene"],
                            }
                        ],
                    }
                },
            )
        if agent.name == "rag_worker":
            worker_histories.append([message.content for message in session_state.messages])
            worker_request = dict((task_payload or {}).get("worker_request") or {})
            return SimpleNamespace(
                text=f"Evidence for {worker_request.get('task_id')}",
                messages=[RuntimeMessage(role="assistant", content=f"Evidence for {worker_request.get('task_id')}")],
                metadata={},
            )
        if agent.mode == "finalizer":
            assert task_payload is not None
            assert task_payload["task_results"][0]["output"] == "Evidence for task_1"
            return SimpleNamespace(
                text="Final synthesized answer",
                messages=list(session_state.messages),
                metadata={},
            )
        if agent.mode == "verifier":
            return SimpleNamespace(
                text='{"status":"pass","summary":"verified","issues":[],"feedback":""}',
                messages=list(session_state.messages),
                metadata={"verification": {"status": "pass", "summary": "verified", "issues": [], "feedback": ""}},
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Compare the contracts and synthesize the answer.",
        callbacks=[],
    )

    assert result.text == "Final synthesized answer"
    assert worker_histories == [[]]


def test_spawn_worker_tool_blocks_background_launch_for_agents_that_disallow_it(tmp_path: Path) -> None:
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    tool_context = ToolContext(
        settings=kernel.settings,
        providers=None,
        stores=None,
        session=session_state,
        paths=kernel.paths,
        active_definition=coordinator,
    )

    result = kernel.spawn_worker_from_tool(
        tool_context,
        prompt="Create a plan",
        agent_name="planner",
        description="planner task",
        run_in_background=True,
    )

    assert "does not allow background jobs" in str(result.get("error") or "")


def test_coordinator_revises_final_answer_when_verifier_requests_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    finalizer_calls = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Compute answer",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the main answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            return SimpleNamespace(
                text="Draft worker output",
                messages=[RuntimeMessage(role="assistant", content="Draft worker output")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            answer = "Initial answer" if len(finalizer_calls) == 1 else "Revised answer"
            return SimpleNamespace(text=answer, messages=list(session_state.messages), metadata={})
        if agent.mode == "verifier":
            return SimpleNamespace(
                text='{"status":"revise","summary":"needs caveat","issues":["missing caveat"],"feedback":"Add the missing caveat."}',
                messages=list(session_state.messages),
                metadata={
                    "verification": {
                        "status": "revise",
                        "summary": "needs caveat",
                        "issues": ["missing caveat"],
                        "feedback": "Add the missing caveat.",
                    }
                },
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Handle this carefully.",
        callbacks=[],
    )

    assert result.text == "Revised answer"
    assert len(finalizer_calls) == 2
    assert finalizer_calls[-1]["verification"]["status"] == "revise"
