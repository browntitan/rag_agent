from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.memory.context_builder import MemoryContextBuilder
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.runtime.context import RuntimePaths


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def _session() -> SessionState:
    return SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
    )


def test_file_memory_store_writes_authoritative_and_derived_files(tmp_path: Path) -> None:
    store = FileMemoryStore(_paths(tmp_path))
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="conversation",
        key="preferred_name",
        value="Shiv",
    )

    memory_dir = _paths(tmp_path).conversation_memory_dir("tenant", "user", "conversation")
    index_payload = json.loads((memory_dir / "index.json").read_text(encoding="utf-8"))
    assert index_payload["entries"]["preferred_name"]["value"] == "Shiv"
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert list((memory_dir / "topics").glob("*.md"))


def test_memory_context_builder_and_extractor_use_scopes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()

    saved = extractor.apply_from_text(
        session,
        "preferred_name: Shiv\nfavorite_editor: Neovim",
        scopes=["user"],
    )
    assert saved == 2

    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        memory_scopes=["user"],
    )
    context = MemoryContextBuilder(store, max_chars=500).build_for_agent(agent, session)
    assert "preferred_name" in context
    assert "favorite_editor" in context
    assert "conversation memory" not in context


def test_memory_extractor_can_apply_from_recent_messages(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()
    session.append_message("user", "remember that favorite_language is Python.")
    session.append_message("assistant", "Noted.")

    saved = extractor.apply_from_messages(session, session.messages, scopes=["user"])

    assert saved == 1
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="favorite_language",
        )
        == "Python"
    )


def test_memory_extractor_parses_key_value_assignments_from_sentences(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()

    saved = extractor.apply_from_text(
        session,
        "Remember these exact values: risk_reserve_monthly_usd=40250; target_jurisdiction=England and Wales.",
        scopes=["user"],
    )

    assert saved == 2
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="risk_reserve_monthly_usd",
        )
        == "40250"
    )
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="target_jurisdiction",
        )
        == "England and Wales"
    )


def test_memory_context_builder_respects_scope_order_and_char_budget(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    session = _session()
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="conversation",
        key="project_status",
        value="active",
    )
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="user",
        key="preferred_name",
        value="Shiv",
    )
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        memory_scopes=["conversation", "user"],
    )

    context = MemoryContextBuilder(store, max_chars=35).build_for_agent(agent, session)

    assert context.startswith("[conversation memory]")
    assert "project_status" in context
    assert "preferred_name" not in context


def test_file_memory_store_serializes_parallel_writes_to_same_scope(tmp_path: Path) -> None:
    store = FileMemoryStore(_paths(tmp_path))

    def save_pair(index: int) -> None:
        store.save(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key=f"key_{index}",
            value=f"value_{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save_pair, range(8)))

    keys = store.list_keys(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="user",
    )
    assert keys == [f"key_{index}" for index in range(8)]
