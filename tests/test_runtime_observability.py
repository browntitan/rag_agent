from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agentic_chatbot.agents.session import ChatSession
from agentic_chatbot_next.app.service import AppContext, RuntimeService
from agentic_chatbot_next.observability.callbacks import RuntimeTraceCallbackHandler
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


def _runtime_settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        agents_dir=repo_root / "data" / "agents",
        max_worker_concurrency=2,
        enable_coordinator_mode=False,
        runtime_events_enabled=True,
        planner_max_tasks=4,
        llm_router_enabled=False,
        llm_router_confidence_threshold=0.70,
        workspace_dir=None,
        clear_scratchpad_per_turn=False,
        agent_runtime_mode="planner_executor",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
    )


def test_basic_turn_persists_router_and_basic_runtime_events(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["Hello back from BASIC."]),
        judge=FakeListChatModel(responses=["unused"]),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="basic-conv")

    text = app.process_turn(session, user_text="Hello there")

    assert "Hello back from BASIC" in text
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    event_types = {row["event_type"] for row in events}
    assert {
        "router_decision",
        "basic_turn_started",
        "model_start",
        "model_end",
        "basic_turn_completed",
    }.issubset(event_types)
    router_event = next(row for row in events if row["event_type"] == "router_decision")
    assert router_event["payload"]["route"] == "BASIC"


def test_runtime_trace_callback_records_model_and_tool_events_for_general_agent(tmp_path: Path):
    paths = RuntimePaths(runtime_root=tmp_path / "runtime", workspace_root=tmp_path / "workspaces", memory_root=tmp_path / "memory")
    store = RuntimeTranscriptStore(paths)

    class _Sink:
        def emit(self, event):
            store.append_session_event(event)

    callback = RuntimeTraceCallbackHandler(
        event_sink=_Sink(),
        session_id="tenant:user:general",
        conversation_id="general",
        trace_name="test_general_agent",
        agent_name="general",
        metadata={"route": "AGENT", "router_method": "deterministic"},
    )

    @tool
    def echo_tool(text: str) -> str:
        """Echo the provided text."""
        return f"echo:{text}"

    from agentic_chatbot_next.general_agent import run_general_agent

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"echo_tool","args":{"text":"trace me"},"purpose":"demo"}],"notes":"demo"}',
            "Final answer from plan-execute fallback.",
        ]
    )
    final_text, _, _ = run_general_agent(
        llm,
        tools=[echo_tool],
        messages=[],
        user_text="Use the echo tool.",
        system_prompt="Use tools when needed.",
        callbacks=[callback],
        max_tool_calls=2,
    )

    assert "Final answer" in final_text
    events = store.load_session_events("tenant:user:general")
    event_types = [row["event_type"] for row in events]
    assert "model_start" in event_types
    assert "model_end" in event_types
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    tool_event = next(row for row in events if row["event_type"] == "tool_start")
    assert tool_event["tool_name"] == "echo_tool"


def test_general_agent_renders_rag_tool_output_when_final_ai_message_is_empty(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    class _DummyLLM:
        def bind_tools(self, tools):
            return self

    class _FakeGraph:
        def invoke(self, payload, config=None):
            del payload, config
            return {
                "messages": [
                    AIMessage(content=""),
                    ToolMessage(
                        content=(
                            '{"answer":"Tool-calling reliability improved through stronger retries.",'
                            '"citations":[{"citation_id":"KB_1#chunk0001","doc_id":"KB_1","title":"05_release_notes.md","source_type":"kb","location":"page None","snippet":"retry improvements"}],'
                            '"used_citation_ids":["KB_1#chunk0001"],'
                            '"retrieval_summary":{"query_used":"tool reliability","steps":3,"tool_calls_used":0,"tool_call_log":[],"citations_found":1},'
                            '"followups":[],"warnings":[]}'
                        ),
                        tool_call_id="tool_1",
                    ),
                    AIMessage(content=""),
                ]
            }

    monkeypatch.setattr(
        "langgraph.prebuilt.create_react_agent",
        lambda chat_llm, tools=None: _FakeGraph(),
    )

    final_text, updated_messages, _ = run_general_agent(
        _DummyLLM(),
        tools=[],
        messages=[],
        user_text="Explain the release-note changes with citations.",
        system_prompt="Use tools when needed.",
    )

    assert "Tool-calling reliability improved" in final_text
    assert "Citations:" in final_text
    assert isinstance(updated_messages[-1], AIMessage)
    assert "Citations:" in str(updated_messages[-1].content)


def test_query_loop_forces_plan_execute_for_data_analyst_strategy(monkeypatch):
    captured = {}

    def fake_run_general_agent(*args, **kwargs):
        captured["force_plan_execute"] = kwargs.get("force_plan_execute")
        return "done", [], {"steps": 1, "tool_calls": 0}

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_general_agent", fake_run_general_agent)

    loop = QueryLoop(settings=None, providers=SimpleNamespace(chat=object()), stores=SimpleNamespace())
    agent = AgentDefinition(
        name="data_analyst",
        mode="react",
        prompt_file="data_analyst_agent.md",
        metadata={"execution_strategy": "plan_execute"},
    )
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    session.append_message("user", "Analyze the uploaded CSV files.")
    tool_context = SimpleNamespace(callbacks=[], refresh_from_session_handle=lambda: None)

    result = loop.run(
        agent,
        session,
        user_text="Analyze the uploaded CSV files.",
        tool_context=tool_context,
        tools=[],
    )

    assert result.text == "done"
    assert captured["force_plan_execute"] is True


def test_general_agent_repairs_non_json_plan_output_before_falling_back(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def echo_tool(text: str) -> str:
        """Echo the provided text."""
        return f"echo:{text}"

    class _DummyLLM:
        def bind_tools(self, tools):
            return self

        def __init__(self):
            self.calls = 0

        def invoke(self, messages, config=None):
            del messages, config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="Use echo_tool with text=trace me.")
            if self.calls == 2:
                return AIMessage(
                    content='{"plan":[{"tool":"echo_tool","args":{"text":"trace me"},"purpose":"demo"}],"notes":"repaired"}'
                )
            return AIMessage(content="Final answer after repaired plan.")

    llm = _DummyLLM()
    final_text, _, metadata = run_general_agent(
        llm,
        tools=[echo_tool],
        messages=[],
        user_text="Use the echo tool.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Final answer" in final_text
    assert metadata["fallback"] == "plan_execute"


def test_general_agent_sanitizes_null_tool_args_before_invocation():
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def inspect_like_tool(doc_id: str = "", columns: str = "") -> str:
        """Return the received arguments."""
        return f"{doc_id}|{columns}"

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"inspect_like_tool","args":{"doc_id":null,"columns":null},"purpose":"demo"}],"notes":"demo"}',
            "Final answer after sanitized tool args.",
        ]
    )

    final_text, messages, metadata = run_general_agent(
        llm,
        tools=[inspect_like_tool],
        messages=[],
        user_text="Inspect the uploaded dataset.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Final answer" in final_text
    assert metadata["fallback"] == "plan_execute"
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert tool_messages[-1].content == "|"


def test_data_analyst_plan_execute_falls_back_to_guided_flow_when_execute_code_fails(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["regional_spend.csv","regional_controls.csv"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        return '{"doc_id":"regional_spend.csv","columns":["region","annual_spend_usd","current_reserve_usd"]}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        return '{"region":{"dtype":"object"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        del code, doc_ids
        return '{"success": false, "stdout": "", "stderr": "sandbox failed"}'

    captured = {}

    def fake_guided_fallback(**kwargs):
        captured["called"] = True
        return "guided fallback", [], {"fallback": "data_analyst_guided"}

    monkeypatch.setattr(
        "agentic_chatbot_next.general_agent._run_data_analyst_guided_fallback",
        fake_guided_fallback,
    )

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"load_dataset","args":{},"purpose":"load"},'
            '{"tool":"inspect_columns","args":{"doc_id":null,"columns":null},"purpose":"inspect"},'
            '{"tool":"execute_code","args":{"code":"print(1)","doc_ids":null},"purpose":"run"}],'
            '"notes":"demo"}'
        ]
    )

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code],
        messages=[],
        user_text="Analyze the uploaded CSV files.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert captured["called"] is True
    assert final_text == "guided fallback"
    assert metadata["fallback"] == "data_analyst_guided"


def test_job_runner_builds_runtime_trace_callbacks_for_workers(tmp_path: Path, monkeypatch):
    kernel = RuntimeKernel(
        _runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="original worker prompt",
        session_id="tenant:user:conv",
        description="worker trace propagation",
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
    captured = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        captured["callbacks"] = callbacks
        return SimpleNamespace(
            text="worker complete",
            messages=[],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel._job_runner(job)

    callbacks = captured["callbacks"]
    assert callbacks
    assert any(isinstance(callback, RuntimeTraceCallbackHandler) for callback in callbacks)


def test_ingest_and_summarize_uploads_uses_langchain_callbacks_without_name_error(
    tmp_path: Path,
    monkeypatch,
):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.ingest_paths",
        lambda *args, **kwargs: ["doc-upload-1"],
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="upload-conv")
    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("uploaded content")

    def fake_rag(*, session, query, conversation_context, preferred_doc_ids, callbacks):
        assert preferred_doc_ids == ["doc-upload-1"]
        assert isinstance(callbacks, list)
        return {
            "answer": "Upload summary",
            "citations": [],
            "used_citation_ids": [],
            "warnings": [],
            "followups": [],
        }

    monkeypatch.setattr(app, "_call_rag_direct", fake_rag)

    doc_ids, rendered = app.ingest_and_summarize_uploads(session, [upload_path])

    assert doc_ids == ["doc-upload-1"]
    assert "Upload summary" in rendered
    assert session.uploaded_doc_ids == ["doc-upload-1"]
