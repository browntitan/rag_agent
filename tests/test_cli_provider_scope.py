from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from agentic_chatbot import cli
from agentic_chatbot.db import vector_schema
from agentic_chatbot_next import rag as next_rag_module


runner = CliRunner()


class _FakeHTTPResponse:
    def __init__(self, *, status_code: int, payload: dict[str, object]):
        self._status_code = status_code
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self._status_code

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_index_skills_uses_store_context_without_full_runtime(monkeypatch):
    captured: dict[str, object] = {}
    settings = SimpleNamespace(default_tenant_id="tenant-123")
    stores = SimpleNamespace(skill_store=object())

    monkeypatch.setattr(cli, "_make_app_or_exit", lambda dotenv=None: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(
        cli,
        "_make_store_context_or_exit",
        lambda dotenv=None: cli.StoreContext(settings=settings, stores=stores),
    )

    class FakeSkillIndexSync:
        def __init__(self, settings_arg, stores_arg):
            captured["settings"] = settings_arg
            captured["stores"] = stores_arg

        def sync(self, *, tenant_id):
            captured["tenant_id"] = tenant_id
            return {"indexed": [], "count": 0}

    monkeypatch.setattr(next_rag_module, "SkillIndexSync", FakeSkillIndexSync)

    result = runner.invoke(cli.app, ["index-skills"])

    assert result.exit_code == 0
    assert captured == {
        "settings": settings,
        "stores": stores,
        "tenant_id": "tenant-123",
    }


def test_sync_kb_uses_store_context_without_full_runtime(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    doc_path = kb_dir / "sample.md"
    doc_path.write_text("# Sample\n", encoding="utf-8")

    settings = SimpleNamespace(
        default_tenant_id="tenant-123",
        default_collection_id="default",
        kb_dir=kb_dir,
    )
    stores = SimpleNamespace(doc_store=object())

    monkeypatch.setattr(cli, "_make_app_or_exit", lambda dotenv=None: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(
        cli,
        "_make_store_context_or_exit",
        lambda dotenv=None: cli.StoreContext(settings=settings, stores=stores),
    )

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id):
        captured["settings"] = settings_arg
        captured["stores"] = stores_arg
        captured["paths"] = [str(path) for path in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-001"]

    monkeypatch.setattr(next_rag_module, "ingest_paths", fake_ingest_paths)

    result = runner.invoke(cli.app, ["sync-kb"])

    assert result.exit_code == 0
    assert captured == {
        "settings": settings,
        "stores": stores,
        "paths": [str(doc_path)],
        "source_type": "kb",
        "tenant_id": "tenant-123",
        "collection_id": "default",
    }


def _doctor_settings():
    return SimpleNamespace(
        llm_provider="ollama",
        judge_provider="ollama",
        embeddings_provider="ollama",
        embedding_dim=768,
        pg_dsn="postgresql://user:pass@localhost:5432/ragdb",
        http2_enabled=True,
        ssl_verify=True,
        ssl_cert_file=None,
        tiktoken_enabled=False,
        tiktoken_cache_dir=None,
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="qwen3:8b",
        ollama_judge_model="qwen3:8b",
        ollama_embed_model="nomic-embed-text",
    )


def test_doctor_fails_when_selected_ollama_model_is_missing(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "qwen3:8b"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 1
    assert "nomic-embed-text" in result.output


def test_doctor_passes_when_selected_ollama_models_exist(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 0
    assert "Doctor checks passed." in result.output
    assert "nomic-embed-text" in result.output


class _FakeDbCursor:
    def __init__(self, dimensions: dict[str, int | None], executed: list[tuple[str, object]] | None = None):
        self.dimensions = dimensions
        self.executed = executed if executed is not None else []
        self.table_name: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if params:
            self.table_name = params[0]

    def fetchone(self):
        if self.table_name is None:
            return None
        dim = self.dimensions.get(self.table_name)
        if dim is None:
            return None
        return (f"vector({dim})",)


class _FakeDbConnection:
    def __init__(self, dimensions: dict[str, int | None], executed: list[tuple[str, object]] | None = None):
        self.dimensions = dimensions
        self.executed = executed if executed is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeDbCursor(self.dimensions, self.executed)

    def commit(self):
        return None


def test_doctor_fails_when_skill_chunks_dimension_is_misaligned(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(
            connect=lambda **kwargs: _FakeDbConnection({"chunks": 768, "skill_chunks": 1536})
        ),
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "skill_chunks.embedding is" in result.output
    assert "EMBEDDING_DIM=768" in result.output
    assert "migrate-embedding-dim --yes" in result.output


def test_migrate_embedding_dim_realigns_chunks_and_skill_chunks(monkeypatch):
    settings = SimpleNamespace(
        embedding_dim=768,
        default_tenant_id="tenant-123",
    )
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: settings)

    import agentic_chatbot.db.connection as db_connection

    apply_calls: list[object] = []
    monkeypatch.setattr(db_connection, "apply_schema", lambda effective_settings: apply_calls.append(effective_settings))
    monkeypatch.setattr(db_connection, "init_pool", lambda effective_settings: None)

    executed: list[tuple[str, object]] = []
    monkeypatch.setattr(db_connection, "get_conn", lambda: _FakeDbConnection({}, executed))

    monkeypatch.setattr(vector_schema, "get_chunks_embedding_dim", lambda: 1536)
    monkeypatch.setattr(vector_schema, "get_skill_chunks_embedding_dim", lambda: 1536)

    changed: list[tuple[str, int]] = []
    monkeypatch.setattr(
        vector_schema,
        "set_chunks_embedding_dim",
        lambda target_dim: changed.append(("chunks", target_dim)) or True,
    )
    monkeypatch.setattr(
        vector_schema,
        "set_skill_chunks_embedding_dim",
        lambda target_dim: changed.append(("skill_chunks", target_dim)) or True,
    )

    result = runner.invoke(cli.app, ["migrate-embedding-dim", "--yes", "--skip-reindex-kb"])

    assert result.exit_code == 0
    assert ("chunks", 768) in changed
    assert ("skill_chunks", 768) in changed
    assert any("TRUNCATE TABLE skill_chunks, skills, chunks, documents" in query for query, _ in executed)
    assert len(apply_calls) == 2
