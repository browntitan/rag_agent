from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_demo_notebook.lib.scenario_runner import (  # noqa: E402
    REQUIRED_AGENT_COVERAGE,
    ScenarioDefinition,
    ScenarioRunner,
    _ensure_repo_import_roots,
    load_scenarios,
    validate_agent_coverage,
)
from new_demo_notebook.lib.preflight import run_preflight  # noqa: E402
from new_demo_notebook.lib.server import BackendServerManager  # noqa: E402
from new_demo_notebook.lib.client import GatewayClient  # noqa: E402
from new_demo_notebook.lib.trace_reader import (  # noqa: E402
    cleanup_conversation_artifacts,
    collect_trace_bundle,
    extract_observed_agents,
    extract_observed_route,
)


def test_server_manager_starts_waits_and_stops(monkeypatch, tmp_path: Path):
    started = {}

    class DummyProcess:
        def __init__(self):
            self._returncode = None

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._returncode = -9

    def fake_popen(command, cwd, stdout, stderr, text, start_new_session):
        started["command"] = command
        started["cwd"] = cwd
        started["stdout"] = stdout
        return DummyProcess()

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"status": "ready"}

    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", fake_popen)
    monkeypatch.setattr("new_demo_notebook.lib.server.httpx.get", lambda *args, **kwargs: DummyResponse())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    base_url = manager.start(timeout_seconds=0.1)
    manager.stop()

    assert base_url == "http://127.0.0.1:8765"
    assert started["command"][1:4] == ["run.py", "serve-api", "--host"]
    assert manager.log_path.name == "server.log"


def test_server_manager_uses_env_ready_timeout_by_default(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setenv("NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS", "321")
    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    monkeypatch.setattr(
        manager,
        "wait_until_ready",
        lambda *, timeout_seconds=None: captured.setdefault("timeout_seconds", timeout_seconds),
    )

    manager.start()
    manager.stop()

    assert captured["timeout_seconds"] == 321.0


def test_server_manager_explicit_timeout_overrides_env(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setenv("NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS", "321")
    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    monkeypatch.setattr(
        manager,
        "wait_until_ready",
        lambda *, timeout_seconds=None: captured.setdefault("timeout_seconds", timeout_seconds),
    )

    manager.start(timeout_seconds=12.5)
    manager.stop()

    assert captured["timeout_seconds"] == 12.5


def test_gateway_client_uses_extended_default_timeout(monkeypatch):
    monkeypatch.delenv("NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS", raising=False)

    client = GatewayClient("http://127.0.0.1:9999")
    try:
        assert client._client.timeout.read == 180.0
        assert client._client.timeout.connect == 10.0
    finally:
        client.close()


def test_trace_reader_merges_session_and_job_artifacts(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    session_dir = runtime_root / "sessions" / "tenant:user:demo-conv"
    job_dir = runtime_root / "jobs" / "job_demo"
    session_dir.mkdir(parents=True)
    (job_dir / "artifacts").mkdir(parents=True)
    workspace = workspace_root / "tenant:user:demo-conv"
    workspace.mkdir(parents=True)
    (workspace / ".meta").write_text(json.dumps({"session_id": "tenant:user:demo-conv"}), encoding="utf-8")

    (session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-conv", "conversation_id": "demo-conv"}),
        encoding="utf-8",
    )
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"kind": "message", "message": {"role": "user", "content": "hello"}}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "router_decision",
                "session_id": "tenant:user:demo-conv",
                "created_at": "2026-01-01T00:00:00+00:00",
                "agent_name": "router",
                "payload": {"conversation_id": "demo-conv", "route": "BASIC"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_demo",
                "session_id": "tenant:user:demo-conv",
                "agent_name": "memory_maintainer",
                "status": "completed",
                "prompt": "remember this",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "job_completed",
                "session_id": "tenant:user:demo-conv",
                "job_id": "job_demo",
                "created_at": "2026-01-01T00:00:01+00:00",
                "agent_name": "memory_maintainer",
                "payload": {"conversation_id": "demo-conv"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-conv")

    assert bundle.session_ids == ["tenant:user:demo-conv"]
    assert extract_observed_route(bundle) == "BASIC"
    assert "memory_maintainer" in extract_observed_agents(bundle)
    assert bundle.workspace_files["tenant:user:demo-conv"] == []

    cleanup_conversation_artifacts(runtime_root, workspace_root, "demo-conv")
    assert not session_dir.exists()
    assert not job_dir.exists()
    assert not workspace.exists()


def test_trace_reader_filters_jobs_to_matching_conversation(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    matching_session_dir = runtime_root / "sessions" / "tenant-user-demo-a"
    other_session_dir = runtime_root / "sessions" / "tenant-user-demo-b"
    matching_session_dir.mkdir(parents=True)
    other_session_dir.mkdir(parents=True)

    (matching_session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-a", "conversation_id": "demo-a"}),
        encoding="utf-8",
    )
    (other_session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-b", "conversation_id": "demo-b"}),
        encoding="utf-8",
    )

    matching_job_dir = runtime_root / "jobs" / "job_match"
    unrelated_job_dir = runtime_root / "jobs" / "job_other"
    (matching_job_dir / "artifacts").mkdir(parents=True)
    (unrelated_job_dir / "artifacts").mkdir(parents=True)
    (matching_job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_match",
                "session_id": "tenant:user:demo-a",
                "agent_name": "utility",
                "status": "completed",
                "metadata": {"session_state": {"conversation_id": "demo-a"}},
            }
        ),
        encoding="utf-8",
    )
    (unrelated_job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_other",
                "session_id": "tenant:user:demo-b",
                "agent_name": "utility",
                "status": "completed",
                "metadata": {"session_state": {"conversation_id": "demo-b"}},
            }
        ),
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-a")

    assert [job["job_id"] for job in bundle.jobs] == ["job_match"]


def test_scenario_runner_reuses_same_conversation_scope_for_ingest_and_chat(tmp_path: Path):
    calls = {"ingest": [], "chat": []}

    class FakeClient:
        def ingest(self, *, paths, conversation_id, source_type="upload", request_id=""):
            calls["ingest"].append((tuple(paths), conversation_id, source_type))
            return {"doc_ids": ["doc-1"]}

        def chat_turn(self, *, history, user_text, conversation_id, model, force_agent=False, request_id="", metadata=None):
            calls["chat"].append((conversation_id, user_text, force_agent))
            return type("Resp", (), {"text": f"reply:{user_text}"})()

    scenario = ScenarioDefinition.from_dict(
        {
            "id": "test-scenario",
            "title": "Test Scenario",
            "description": "desc",
            "conversation_id": "demo-conv",
            "ingest_paths": ["new_demo_notebook/demo_data/regional_spend.csv"],
            "messages": ["Turn one", "Turn two"],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["general"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["general"],
        }
    )

    def fake_trace_loader(runtime_root, workspace_root, conversation_id):
        from new_demo_notebook.lib.trace_reader import TraceBundle

        return TraceBundle(
            conversation_id=conversation_id,
            session_ids=["tenant:user:demo-conv"],
            event_rows=[
                {
                    "event_type": "router_decision",
                    "session_id": "tenant:user:demo-conv",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "agent_name": "router",
                    "job_id": "",
                    "tool_name": "",
                    "route": "AGENT",
                    "router_method": "deterministic",
                    "suggested_agent": "",
                    "conversation_id": conversation_id,
                    "payload": {"conversation_id": conversation_id, "route": "AGENT"},
                },
                {
                    "event_type": "agent_turn_completed",
                    "session_id": "tenant:user:demo-conv",
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "agent_name": "general",
                    "job_id": "",
                    "tool_name": "",
                    "route": "AGENT",
                    "router_method": "deterministic",
                    "suggested_agent": "",
                    "conversation_id": conversation_id,
                    "payload": {"conversation_id": conversation_id, "route": "AGENT"},
                },
            ],
            jobs=[],
            job_events=[],
        )

    runner = ScenarioRunner(
        client=FakeClient(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
        trace_loader=fake_trace_loader,
        cleanup_fn=lambda runtime_root, workspace_root, conversation_id: None,
    )

    result = runner.run_scenario(scenario)

    assert result.success is True
    assert calls["ingest"][0][1] == "demo-conv"
    assert [conversation_id for conversation_id, _, _ in calls["chat"]] == ["demo-conv", "demo-conv"]


def test_extract_observed_agents_includes_coordinator_phase_agents_from_event_payload(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    session_dir = runtime_root / "sessions" / "tenant-user-demo-coordinator"
    session_dir.mkdir(parents=True)
    (session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-coordinator", "conversation_id": "demo-coordinator"}),
        encoding="utf-8",
    )
    (session_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "coordinator_planning_started",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "planner_agent": "planner",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "coordinator_finalizer_completed",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:01+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "finalizer_agent": "finalizer",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "coordinator_verifier_completed",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:02+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "verifier_agent": "verifier",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-coordinator")
    observed = extract_observed_agents(bundle)

    assert "coordinator" in observed
    assert "planner" in observed
    assert "finalizer" in observed
    assert "verifier" in observed


def test_scenario_runner_recognizes_drained_task_notifications_in_session_state(tmp_path: Path):
    scenario = ScenarioDefinition.from_dict(
        {
            "id": "memory_maintainer_background",
            "title": "Memory Maintainer Background Job",
            "description": "desc",
            "conversation_id": "demo-memory",
            "ingest_paths": [],
            "messages": ["launch worker", "summarize worker"],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["general", "memory_maintainer"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["memory_maintainer"],
        }
    )
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    bundle = collect_trace_bundle(tmp_path / "runtime", tmp_path / "workspaces", "missing")
    bundle = bundle.__class__(
        conversation_id="demo-memory",
        session_states=[
            {
                "pending_notifications": [
                    {
                        "job_id": "job_123",
                        "status": "completed",
                        "summary": "Saved 3 memory entries.",
                        "metadata": {"agent_name": "memory_maintainer"},
                    }
                ],
                "messages": [
                    {
                        "role": "system",
                        "metadata": {
                            "notification": {
                                "job_id": "job_123",
                                "status": "completed",
                                "summary": "Saved 3 memory entries.",
                            }
                        },
                    }
                ],
            }
        ],
        transcript_rows=[
            {
                "kind": "notification",
                "notification": {
                    "job_id": "job_123",
                    "status": "completed",
                    "summary": "Saved 3 memory entries.",
                },
            }
        ],
        notifications=[],
        jobs=[
            {
                "job_id": "job_123",
                "agent_name": "memory_maintainer",
                "status": "completed",
            }
        ],
    )

    runner._load_memory_snapshot = lambda bundle, conversation_id: {  # type: ignore[method-assign]
        "conversation": {"demo_owner": "Platform Reliability"},
        "user": {
            "launch_window": "Q3 2026",
            "reserve_strategy": "Board review pending",
        },
    }

    errors = runner._scenario_specific_errors(
        scenario,
        outputs=["Worker completed and stored the keys."],
        raw_responses=[],
        bundle=bundle,
    )

    assert errors == []


def test_scenario_runner_bootstrap_adds_src_root_to_sys_path(monkeypatch):
    repo_root = REPO_ROOT
    src_root = repo_root / "src"
    custom_sys_path = [entry for entry in sys.path if entry not in {str(repo_root), str(src_root)}]
    monkeypatch.setattr(sys, "path", custom_sys_path)

    _ensure_repo_import_roots()

    assert str(src_root) in sys.path
    assert str(repo_root) in sys.path


def test_scenario_manifest_covers_all_required_agents():
    scenarios = load_scenarios(REPO_ROOT / "new_demo_notebook" / "scenarios" / "scenarios.json")
    coverage = validate_agent_coverage(scenarios, required_agents=REQUIRED_AGENT_COVERAGE)

    assert set(REQUIRED_AGENT_COVERAGE).issubset(set(coverage))


def test_preflight_reports_missing_docker_and_unreachable_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PG_DSN", "postgresql://localhost:59999/ragdb")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "azure")
    monkeypatch.setenv("JUDGE_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setattr("new_demo_notebook.lib.preflight.shutil.which", lambda name: None)

    report = run_preflight(
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )

    rows = {row["name"]: row for row in report.to_rows()}
    assert report.ready is False
    assert rows["docker"]["ok"] is False
    assert rows["database"]["ok"] is False
    assert rows["chat_provider"]["ok"] is False
