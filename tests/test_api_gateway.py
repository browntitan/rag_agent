"""Tests for the OpenAI-compatible FastAPI gateway."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentic_chatbot.api import main as api_main
from agentic_chatbot_next.runtime.context import filesystem_key


class DummyBot:
    def __init__(self, answer: str = "stubbed answer"):
        self.answer = answer
        self.calls: list[dict[str, object]] = []
        self.ctx = SimpleNamespace(stores=object())

    def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False):
        self.calls.append(
            {
                "session": session,
                "user_text": user_text,
                "upload_paths": list(upload_paths or []),
                "force_agent": force_agent,
            }
        )
        return self.answer


def _make_settings(tmp_path: Path):
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir()
    return SimpleNamespace(
        gateway_model_id="enterprise-agent",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
        workspace_dir=workspace_dir,
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
    client, _, _ = _make_client(tmp_path)
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
