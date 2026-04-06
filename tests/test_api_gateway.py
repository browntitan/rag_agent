"""Tests for the OpenAI-compatible FastAPI gateway."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentic_chatbot_next.api import main as api_main
from agentic_chatbot_next.rag.ingest import KBCoverageStatus
from agentic_chatbot_next.runtime.context import filesystem_key


class DummyBot:
    def __init__(self, answer: str = "stubbed answer"):
        self.answer = answer
        self.calls: list[dict[str, object]] = []
        self.ctx = SimpleNamespace(stores=object())
        self.kb_status = KBCoverageStatus(
            tenant_id="local-dev",
            collection_id="default",
            configured_source_paths=(),
            missing_source_paths=(),
            indexed_source_paths=(),
            indexed_doc_count=0,
        )

    def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, extra_callbacks=None):
        self.calls.append(
            {
                "session": session,
                "user_text": user_text,
                "upload_paths": list(upload_paths or []),
                "force_agent": force_agent,
                "extra_callbacks": list(extra_callbacks or []),
            }
        )
        return self.answer

    def get_kb_status(self, tenant_id=None, *, refresh=False, attempt_sync=False):
        del tenant_id, refresh, attempt_sync
        return self.kb_status


def _make_settings(tmp_path: Path):
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir()
    return SimpleNamespace(
        gateway_model_id="enterprise-agent",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
        workspace_dir=workspace_dir,
        agent_chat_model_overrides={"general": "gpt-oss:20b"},
        agent_judge_model_overrides={"general": "gpt-oss:20b"},
    )


def _make_client(tmp_path: Path):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    return client, bot, settings


def _clear_overrides() -> None:
    api_main.app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_models_returns_gateway_model(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/v1/models")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "enterprise-agent"


@pytest.mark.asyncio
async def test_health_ready_returns_200_when_kb_is_ready(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/health/ready")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": "enterprise-agent"}


@pytest.mark.asyncio
async def test_health_ready_returns_503_when_kb_coverage_is_missing(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    bot.kb_status = KBCoverageStatus(
        tenant_id="local-dev",
        collection_id="default",
        configured_source_paths=("/tmp/docs/ARCHITECTURE.md",),
        missing_source_paths=("/tmp/docs/ARCHITECTURE.md",),
        indexed_source_paths=(),
        indexed_doc_count=0,
        sync_attempted=True,
    )
    try:
        response = await client.get("/health/ready")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["reason"] == "kb_coverage_missing"
    assert payload["collection_id"] == "default"
    assert payload["suggested_fix"] == "python run.py sync-kb --collection-id default"


@pytest.mark.asyncio
async def test_list_models_ignores_agent_specific_runtime_model_overrides(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/v1/models")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "enterprise-agent"


@pytest.mark.asyncio
async def test_chat_completions_uses_client_history_and_conversation_scope(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "chat-001"},
            json={
                "model": "enterprise-agent",
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": [{"type": "text", "text": "Summarize the auth doc."}]},
                ],
                "metadata": {"force_agent": True},
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "stubbed answer"

    call = bot.calls[0]
    session = call["session"]
    assert call["user_text"] == "Summarize the auth doc."
    assert call["upload_paths"] == []
    assert call["force_agent"] is True
    assert session.conversation_id == "chat-001"
    assert session.session_id == "local-dev:local-cli:chat-001"
    assert [msg.content for msg in session.messages] == [
        "You are concise.",
        "How can I help?",
    ]


@pytest.mark.asyncio
async def test_chat_completions_reuses_same_session_scope_for_repeated_conversation_id(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        for question in ("First turn", "Second turn"):
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-Conversation-ID": "stable-scope"},
                json={
                    "model": "enterprise-agent",
                    "messages": [{"role": "user", "content": question}],
                },
            )
            assert response.status_code == 200
    finally:
        await client.aclose()
        _clear_overrides()

    session_ids = [call["session"].session_id for call in bot.calls]
    assert session_ids == [
        "local-dev:local-cli:stable-scope",
        "local-dev:local-cli:stable-scope",
    ]


@pytest.mark.asyncio
async def test_chat_completions_streaming_returns_sse_chunks(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object": "chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text
    assert len(bot.calls) == 1
    assert len(bot.calls[0]["extra_callbacks"]) == 1


def test_stream_with_progress_waits_for_slow_basic_turn_without_progress_events(monkeypatch):
    real_thread_cls = threading.Thread

    class _Events:
        def __init__(self, callback):
            self._callback = callback

        def get(self, timeout=None):
            del timeout
            if self._callback.done:
                return None
            raise queue.Empty

    class _FakeProgressCallback:
        def __init__(self):
            self.done = False
            self.events = _Events(self)

        def mark_done(self):
            self.done = True

    class _JoinlessThread:
        def __init__(self, target, daemon=False):
            self._thread = real_thread_cls(target=target, daemon=daemon)

        def start(self):
            self._thread.start()

        def is_alive(self):
            return self._thread.is_alive()

        def join(self, timeout=None):
            del timeout
            return None

    class _SlowBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, extra_callbacks=None):
            del session, user_text, upload_paths, force_agent, extra_callbacks
            threading.Event().wait(0.05)
            return "Late answer"

    monkeypatch.setattr(api_main, "ProgressCallback", _FakeProgressCallback)
    monkeypatch.setattr(api_main.threading, "Thread", _JoinlessThread)

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_SlowBot(),
            force_agent=False,
            prompt_tokens=1,
        )
    )

    assert '"content": "Late answer"' in payload
    assert "data: [DONE]" in payload


@pytest.mark.asyncio
async def test_chat_completions_rejects_unknown_model(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model: wrong-model"


@pytest.mark.asyncio
async def test_chat_completions_requires_last_message_to_be_user(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == "last message must have role='user'"


@pytest.mark.asyncio
async def test_ingest_documents_indexes_files_and_copies_them_into_existing_workspace(tmp_path, monkeypatch):
    client, bot, settings = _make_client(tmp_path)
    src = tmp_path / "sales.csv"
    src.write_text("region,revenue\nNA,100\n")
    session_workspace = settings.workspace_dir / filesystem_key("local-dev:local-cli:conv-007")

    captured: dict[str, object] = {}

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id):
        captured["settings"] = settings_arg
        captured["stores"] = stores_arg
        captured["paths"] = [str(path) for path in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        return ["doc-upload-1"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)

    try:
        response = await client.post(
            "/v1/ingest/documents",
            headers={"X-Conversation-ID": "conv-007"},
            json={
                "paths": [str(src)],
                "source_type": "upload",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_ids"] == ["doc-upload-1"]
    assert payload["workspace_copies"] == ["sales.csv"]
    assert (session_workspace / "sales.csv").read_text() == "region,revenue\nNA,100\n"
    assert captured == {
        "settings": settings,
        "stores": bot.ctx.stores,
        "paths": [str(src)],
        "source_type": "upload",
        "tenant_id": "local-dev",
    }
