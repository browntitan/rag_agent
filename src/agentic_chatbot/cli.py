from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.demo import (
    DemoScenario,
    DemoTurn,
    evaluate_response,
    load_demo_scenarios,
    render_scenario_summary,
)
from agentic_chatbot.providers import (
    ProviderConfigurationError,
    ProviderDependencyError,
    build_embeddings,
    build_providers,
    validate_provider_configuration,
    validate_provider_dependencies,
)
from agentic_chatbot_next.app.cli_adapter import CliAdapter
from agentic_chatbot_next.app.service import RuntimeService
from agentic_chatbot_next.context import build_local_context
from agentic_chatbot_next.session import ChatSession


app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class DoctorCheckResult:
    name: str
    status: str
    details: str
    remediation: str = ""


@dataclass(frozen=True)
class StoreContext:
    settings: Settings
    stores: Any


def _build_bot(settings: Settings) -> RuntimeService:
    providers = build_providers(settings)
    return CliAdapter.create_service(settings, providers)


def _build_store_context(settings: Settings) -> StoreContext:
    from agentic_chatbot_next.rag import load_stores

    embeddings = build_embeddings(settings)
    stores = load_stores(settings, embeddings)
    return StoreContext(settings=settings, stores=stores)


def _make_app(dotenv: Optional[str] = None) -> RuntimeService:
    settings = load_settings(dotenv)
    return _build_bot(settings)


def _make_store_context(dotenv: Optional[str] = None) -> StoreContext:
    settings = load_settings(dotenv)
    return _build_store_context(settings)


def _make_local_session(dotenv: Optional[str] = None, conversation_id: Optional[str] = None) -> ChatSession:
    settings = load_settings(dotenv)
    ctx = build_local_context(settings, conversation_id=conversation_id)
    return ChatSession.from_context(ctx)


def _render_demo_notes(scenario_obj: DemoScenario) -> str:
    if not scenario_obj.notes:
        return ""
    return f"Notes:\n{scenario_obj.notes}"


def _coerce_force_agent(global_force_agent: bool, turn: DemoTurn) -> bool:
    if global_force_agent:
        return True
    if turn.force_agent is None:
        return False
    return bool(turn.force_agent)


def _verify_status_style(status: str) -> str:
    style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    return style.get(status, "white")


def _doctor_status_style(status: str) -> str:
    style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
        "SKIP": "cyan",
    }
    return style.get(status, "white")


def _mask_dsn_password(dsn: str) -> str:
    try:
        parsed = urlsplit(dsn)
    except Exception:
        return dsn

    if not parsed.scheme or "@" not in parsed.netloc:
        return dsn

    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    if ":" in userinfo:
        username = userinfo.split(":", 1)[0]
        masked_userinfo = f"{username}:***"
    else:
        masked_userinfo = userinfo

    return urlunsplit((parsed.scheme, f"{masked_userinfo}@{hostinfo}", parsed.path, parsed.query, parsed.fragment))


def _read_table_embedding_dim(conn, table_name: str) -> Optional[int]:
    from agentic_chatbot.db.vector_schema import parse_vector_dimension

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (table_name,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return parse_vector_dimension(str(row[0]))


def _exit_provider_error(exc: Exception) -> None:
    title = "Provider Configuration Error" if isinstance(exc, ProviderConfigurationError) else "Provider Dependency Error"
    console.print(Panel(str(exc), title=title, border_style="red"))
    console.print("Run `python run.py doctor` to validate providers, database, and connectivity checks.")
    raise typer.Exit(code=2)


def _make_app_or_exit(dotenv: Optional[str] = None) -> RuntimeService:
    try:
        return _make_app(dotenv)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _build_bot_or_exit(settings: Settings) -> RuntimeService:
    try:
        return _build_bot(settings)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _make_store_context_or_exit(dotenv: Optional[str] = None) -> StoreContext:
    try:
        return _make_store_context(dotenv)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _build_store_context_or_exit(settings: Settings) -> StoreContext:
    try:
        return _build_store_context(settings)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _with_demo_settings(settings: Settings) -> Settings:
    if settings.llm_provider.lower() != "ollama":
        return settings
    target_predict = max(settings.ollama_num_predict, settings.demo_ollama_num_predict)
    if target_predict == settings.ollama_num_predict:
        return settings
    return replace(settings, ollama_num_predict=target_predict)


def _selected_ollama_models(settings: Settings) -> dict[str, str]:
    models: dict[str, str] = {}
    if settings.llm_provider.lower() == "ollama":
        models["llm"] = settings.ollama_chat_model
    if settings.judge_provider.lower() == "ollama":
        models["judge"] = settings.ollama_judge_model
    if settings.embeddings_provider.lower() == "ollama":
        models["embeddings"] = settings.ollama_embed_model
    return models


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    force_agent: bool = typer.Option(False, "--force-agent"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a single-turn query."""

    bot = _make_app_or_exit(dotenv)
    session = _make_local_session(dotenv)

    response = bot.process_turn(session, user_text=question, upload_paths=upload, force_agent=force_agent)

    console.print(Panel(response, title="Assistant"))


@app.command()
def chat(
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Start an interactive chat session. Use /upload PATH to ingest docs mid-chat."""

    bot = _make_app_or_exit(dotenv)
    session = _make_local_session(dotenv)

    if upload:
        console.print("[bold]Ingesting uploads...[/bold]")
        bot.ingest_and_summarize_uploads(session, upload)

    console.print(Panel("Type your message. Commands: /upload PATH, /exit", title="Agentic Chatbot"))

    while True:
        try:
            user_text = console.input("\n[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break

        if user_text.startswith("/upload"):
            parts = user_text.split(maxsplit=1)
            if len(parts) < 2:
                console.print("Usage: /upload PATH")
                continue
            path = Path(parts[1]).expanduser()
            if not path.exists():
                console.print(f"File not found: {path}")
                continue
            doc_ids, summary = bot.ingest_and_summarize_uploads(session, [path])
            console.print(Panel(summary, title=f"Ingested: {doc_ids}"))
            continue

        response = bot.process_turn(session, user_text=user_text, upload_paths=[])
        console.print(Panel(response, title="Assistant"))


@app.command()
def init_kb(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Deprecated alias: seed the built-in demo KB into the default collection."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    console.print(
        "[yellow]`init-kb` is deprecated.[/yellow] "
        "Use `python run.py sync-kb` for normal DB-first ingestion. "
        "This command now only seeds the bundled demo KB."
    )
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    kb_paths = sorted(Path(store_ctx.settings.kb_dir).glob("*"))
    ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        kb_paths,
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=store_ctx.settings.default_collection_id,
    )
    records = store_ctx.stores.doc_store.list_documents(
        tenant_id=tenant_id,
        source_type="kb",
        collection_id=store_ctx.settings.default_collection_id,
    )
    docs = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "source_type": r.source_type,
            "collection_id": r.collection_id,
            "num_chunks": r.num_chunks,
            "doc_structure_type": r.doc_structure_type,
        }
        for r in records
    ]
    console.print(json.dumps(docs, indent=2, ensure_ascii=False)[:4000])


@app.command("sync-kb")
def sync_kb(
    path: List[Path] = typer.Option([], "--path", "-p", help="File(s) to ingest. Defaults to all files in data/kb."),
    source_type: str = typer.Option("kb", "--source-type"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Ingest a corpus into PostgreSQL + pgvector using an explicit collection ID."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    paths = path or sorted(Path(store_ctx.settings.kb_dir).glob("*"))
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    doc_ids = ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        paths,
        source_type=source_type,
        tenant_id=tenant_id,
        collection_id=collection_id or store_ctx.settings.default_collection_id,
    )
    console.print(
        json.dumps(
            {
                "ingested_doc_ids": doc_ids,
                "count": len(doc_ids),
                "collection_id": collection_id or store_ctx.settings.default_collection_id,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("reindex-document")
def reindex_document(
    path: Path = typer.Argument(..., exists=True, readable=True),
    source_type: str = typer.Option("kb", "--source-type"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Delete any existing rows for a source path, then ingest the file again."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    effective_collection = collection_id or store_ctx.settings.default_collection_id
    existing = [
        record
        for record in store_ctx.stores.doc_store.list_documents(tenant_id=tenant_id, collection_id=effective_collection)
        if Path(record.source_path) == path and record.source_type == source_type
    ]
    for record in existing:
        store_ctx.stores.doc_store.delete_document(record.doc_id, tenant_id=tenant_id)

    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    doc_ids = ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        [path],
        source_type=source_type,
        tenant_id=tenant_id,
        collection_id=effective_collection,
    )
    console.print(
        json.dumps(
            {
                "deleted_doc_ids": [record.doc_id for record in existing],
                "ingested_doc_ids": doc_ids,
                "collection_id": effective_collection,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("delete-document")
def delete_document(
    doc_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Delete one indexed document and its chunks from the database."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    record = store_ctx.stores.doc_store.get_document(doc_id, tenant_id=tenant_id)
    if record is None:
        console.print(json.dumps({"deleted": False, "doc_id": doc_id, "reason": "not_found"}, indent=2))
        raise typer.Exit(code=1)
    store_ctx.stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
    console.print(
        json.dumps(
            {
                "deleted": True,
                "doc_id": doc_id,
                "title": record.title,
                "collection_id": record.collection_id,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("index-skills")
def index_skills(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Index repo-authored skill packs into the DB-backed skill store."""

    store_ctx = _make_store_context_or_exit(dotenv)
    from agentic_chatbot_next.rag import SkillIndexSync  # noqa: PLC0415

    result = SkillIndexSync(store_ctx.settings, store_ctx.stores).sync(
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    console.print(json.dumps(result, indent=2, ensure_ascii=False)[:6000])


@app.command("list-skills")
def list_skills(
    agent_scope: str = typer.Option("", "--agent-scope"),
    enabled_only: bool = typer.Option(False, "--enabled-only"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """List DB-indexed skill packs."""

    store_ctx = _make_store_context_or_exit(dotenv)
    records = store_ctx.stores.skill_store.list_skill_packs(
        tenant_id=store_ctx.settings.default_tenant_id,
        agent_scope=agent_scope,
        enabled_only=enabled_only,
    )
    console.print(
        json.dumps(
            [
                {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "agent_scope": record.agent_scope,
                    "tool_tags": record.tool_tags,
                    "task_tags": record.task_tags,
                    "version": record.version,
                    "enabled": record.enabled,
                    "source_path": record.source_path,
                }
                for record in records
            ],
            indent=2,
            ensure_ascii=False,
        )[:6000]
    )


@app.command("inspect-skill")
def inspect_skill(
    skill_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Show one indexed skill pack and its stored chunks."""

    store_ctx = _make_store_context_or_exit(dotenv)
    record = store_ctx.stores.skill_store.get_skill_pack(
        skill_id,
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    if record is None:
        console.print(json.dumps({"found": False, "skill_id": skill_id}, indent=2))
        raise typer.Exit(code=1)
    chunks = store_ctx.stores.skill_store.get_skill_chunks(
        skill_id,
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    console.print(
        json.dumps(
            {
                "found": True,
                "record": {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "agent_scope": record.agent_scope,
                    "tool_tags": record.tool_tags,
                    "task_tags": record.task_tags,
                    "version": record.version,
                    "enabled": record.enabled,
                    "source_path": record.source_path,
                    "description": record.description,
                },
                "chunks": chunks,
            },
            indent=2,
            ensure_ascii=False,
        )[:8000]
    )


@app.command()
def reset_indexes(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Truncate all indexed data from PostgreSQL (documents, chunks, memory, skills)."""

    settings = load_settings(dotenv)

    if not confirm:
        typer.confirm(
            "This will DELETE all documents, chunks, memory, and indexed skills from the database. Continue?",
            abort=True,
        )

    from agentic_chatbot.db.connection import apply_schema, get_conn, init_pool

    init_pool(settings)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory, skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()

    console.print("All indexes cleared. Run sync-kb and index-skills to rebuild.")


@app.command()
def migrate(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Apply the database schema (idempotent — safe to run multiple times)."""

    settings = load_settings(dotenv)
    from agentic_chatbot.db.connection import apply_schema

    apply_schema(settings)
    console.print("Schema applied successfully.")


@app.command("migrate-embedding-dim")
def migrate_embedding_dim(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    target_dim: int = typer.Option(0, "--target-dim", help="Target embedding vector dimension (0 uses EMBEDDING_DIM)."),
    reindex_kb: bool = typer.Option(True, "--reindex-kb/--skip-reindex-kb"),
    reset_memory: bool = typer.Option(False, "--reset-memory/--keep-memory"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Align chunk vector dimension, reset indexed docs/chunks, and optionally reindex KB."""

    settings = load_settings(dotenv)
    desired_dim = target_dim if target_dim > 0 else settings.embedding_dim
    if desired_dim <= 0:
        raise typer.BadParameter("--target-dim must be positive.")

    if not confirm:
        typer.confirm(
            "This will clear indexed documents/chunks to rebuild embeddings at the target dimension. Continue?",
            abort=True,
        )

    from agentic_chatbot.db.connection import apply_schema, get_conn, init_pool
    from agentic_chatbot.db.vector_schema import (
        get_chunks_embedding_dim,
        get_skill_chunks_embedding_dim,
        set_chunks_embedding_dim,
        set_skill_chunks_embedding_dim,
    )

    effective_settings = settings if desired_dim == settings.embedding_dim else replace(settings, embedding_dim=desired_dim)

    apply_schema(effective_settings)
    init_pool(effective_settings)
    before_chunks_dim = get_chunks_embedding_dim()
    before_skill_chunks_dim = get_skill_chunks_embedding_dim()
    if before_chunks_dim is None or before_skill_chunks_dim is None:
        missing = []
        if before_chunks_dim is None:
            missing.append("chunks.embedding")
        if before_skill_chunks_dim is None:
            missing.append("skill_chunks.embedding")
        console.print(
            "[red]Unable to detect "
            + ", ".join(missing)
            + " dimension(s). Ensure schema is applied and the required tables exist.[/red]"
        )
        raise typer.Exit(code=1)
    changed_chunks = set_chunks_embedding_dim(desired_dim)
    changed_skill_chunks = set_skill_chunks_embedding_dim(desired_dim)

    with get_conn() as conn:
        with conn.cursor() as cur:
            if reset_memory:
                cur.execute("TRUNCATE TABLE memory, skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
            else:
                cur.execute("TRUNCATE TABLE skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()

    # Recreate dropped vector index(es) if the embedding column type was altered.
    apply_schema(effective_settings)

    after_chunks_dim = get_chunks_embedding_dim()
    after_skill_chunks_dim = get_skill_chunks_embedding_dim()
    console.print(
        "Embedding schema alignment complete "
        "(chunks: "
        f"{before_chunks_dim}->{after_chunks_dim}, "
        "skill_chunks: "
        f"{before_skill_chunks_dim}->{after_skill_chunks_dim}, "
        f"schema_changed={'yes' if changed_chunks or changed_skill_chunks else 'no'})."
    )

    if desired_dim != settings.embedding_dim:
        console.print(
            f"[yellow]Note:[/yellow] Settings currently use EMBEDDING_DIM={settings.embedding_dim}. "
            f"Update `.env` to EMBEDDING_DIM={desired_dim} before normal runs."
        )

    if reindex_kb:
        store_ctx = _build_store_context_or_exit(effective_settings)
        tenant_id = effective_settings.default_tenant_id
        from agentic_chatbot_next.rag import SkillIndexSync, ingest_paths  # noqa: PLC0415

        kb_paths = sorted(Path(effective_settings.kb_dir).glob("*"))
        ingest_paths(
            effective_settings,
            store_ctx.stores,
            kb_paths,
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_settings.default_collection_id,
        )
        SkillIndexSync(effective_settings, store_ctx.stores).sync(tenant_id=tenant_id)
        kb_docs = store_ctx.stores.doc_store.list_documents(
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_settings.default_collection_id,
        )
        console.print(f"Reindex complete. Demo KB documents: {len(kb_docs)}; skill packs re-synced.")
    else:
        console.print("Skipped KB reindex (--skip-reindex-kb). Run `python run.py sync-kb` and `python run.py index-skills` when ready.")


@app.command()
def doctor(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if any WARN checks are present."),
    timeout_seconds: float = typer.Option(3.0, "--timeout-seconds", min=0.5, help="Timeout used for connectivity checks."),
    check_db: bool = typer.Option(True, "--check-db/--skip-db", help="Enable or skip PostgreSQL connectivity check."),
    check_ollama: bool = typer.Option(
        True,
        "--check-ollama/--skip-ollama",
        help="Enable or skip Ollama API check when Ollama providers are selected.",
    ),
):
    """Run provider/runtime preflight checks for local or Docker execution."""

    settings = load_settings(dotenv)
    provider_set = {
        settings.llm_provider.lower(),
        settings.judge_provider.lower(),
        settings.embeddings_provider.lower(),
    }
    needs_ollama = "ollama" in provider_set
    needs_azure = "azure" in provider_set
    needs_nvidia = "nvidia" in provider_set

    config_lines = [
        f"LLM_PROVIDER={settings.llm_provider}",
        f"JUDGE_PROVIDER={settings.judge_provider}",
        f"EMBEDDINGS_PROVIDER={settings.embeddings_provider}",
        f"EMBEDDING_DIM={settings.embedding_dim}",
        f"PG_DSN={_mask_dsn_password(settings.pg_dsn)}",
        f"HTTP2_ENABLED={settings.http2_enabled}",
        f"SSL_VERIFY={settings.ssl_verify}",
        f"SSL_CERT_FILE={settings.ssl_cert_file or '<unset>'}",
        f"TIKTOKEN_ENABLED={settings.tiktoken_enabled}",
        f"TIKTOKEN_CACHE_DIR={settings.tiktoken_cache_dir or '<unset>'}",
    ]
    if needs_ollama:
        config_lines.append(f"OLLAMA_BASE_URL={settings.ollama_base_url}")
    if needs_azure:
        config_lines.extend(
            [
                f"AZURE_OPENAI_ENDPOINT={settings.azure_openai_endpoint or '<unset>'}",
                f"AZURE_OPENAI_CHAT_DEPLOYMENT={settings.azure_openai_chat_deployment or '<unset>'}",
                f"AZURE_OPENAI_JUDGE_DEPLOYMENT={settings.azure_openai_judge_deployment or '<unset>'}",
                f"AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT={settings.azure_openai_embed_deployment or '<unset>'}",
            ]
        )
    if needs_nvidia:
        config_lines.extend(
            [
                f"NVIDIA_OPENAI_ENDPOINT={settings.nvidia_openai_endpoint or '<unset>'}",
                f"NVIDIA_CHAT_MODEL={settings.nvidia_chat_model or '<unset>'}",
                f"NVIDIA_JUDGE_MODEL={settings.nvidia_judge_model or '<unset>'}",
                f"NVIDIA_API_TOKEN={'<set>' if settings.nvidia_api_token else '<unset>'}",
            ]
        )
    console.print(Panel("\n".join(config_lines), title="Selected Configuration"))

    checks: List[DoctorCheckResult] = []

    issues = validate_provider_dependencies(settings)
    if issues:
        details = "; ".join(f"{issue.module} ({', '.join(issue.contexts)})" for issue in issues)
        remediation = "Install dependencies with `python -m pip install -r requirements.txt`, then rerun `python run.py doctor`."
        checks.append(
            DoctorCheckResult(
                name="Provider dependency packages",
                status="FAIL",
                details=details,
                remediation=remediation,
            )
        )
    else:
        checks.append(
            DoctorCheckResult(
                name="Provider dependency packages",
                status="PASS",
                details="All required provider packages are importable.",
            )
        )

    config_issues = validate_provider_configuration(settings)
    if config_issues:
        details = "; ".join(f"({issue.context}) {issue.message}" for issue in config_issues)
        remediation = (
            "Fix provider variables in .env "
            "(Azure Gov endpoints like https://<resource>.openai.azure.us are supported; "
            "NVIDIA endpoints should be OpenAI-compatible base URLs)."
        )
        checks.append(
            DoctorCheckResult(
                name="Provider runtime configuration",
                status="FAIL",
                details=details,
                remediation=remediation,
            )
        )
    else:
        checks.append(
            DoctorCheckResult(
                name="Provider runtime configuration",
                status="PASS",
                details="Provider env/settings are internally consistent.",
            )
        )

    if check_db:
        try:
            import psycopg2

            with psycopg2.connect(dsn=settings.pg_dsn, connect_timeout=max(1, int(timeout_seconds))) as conn:
                checks.append(
                    DoctorCheckResult(
                        name="PostgreSQL connectivity",
                        status="PASS",
                        details="Connected to PG_DSN successfully.",
                    )
                )
                chunks_dim = _read_table_embedding_dim(conn, "chunks")
                skill_chunks_dim = _read_table_embedding_dim(conn, "skill_chunks")
                if chunks_dim is None or skill_chunks_dim is None:
                    missing = []
                    if chunks_dim is None:
                        missing.append("chunks.embedding")
                    if skill_chunks_dim is None:
                        missing.append("skill_chunks.embedding")
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="WARN",
                            details=(
                                "Could not detect "
                                + ", ".join(missing)
                                + " dimension(s) (table may not exist yet)."
                            ),
                            remediation="Run `python run.py migrate` first, then rerun doctor.",
                        )
                    )
                elif chunks_dim != settings.embedding_dim or skill_chunks_dim != settings.embedding_dim:
                    mismatches = []
                    if chunks_dim != settings.embedding_dim:
                        mismatches.append(
                            f"chunks.embedding is vector({chunks_dim})"
                        )
                    if skill_chunks_dim != settings.embedding_dim:
                        mismatches.append(
                            f"skill_chunks.embedding is vector({skill_chunks_dim})"
                        )
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="FAIL",
                            details=(
                                "; ".join(mismatches)
                                + f" but EMBEDDING_DIM={settings.embedding_dim}."
                            ),
                            remediation="Run `python run.py migrate-embedding-dim --yes` to realign and rebuild vectors.",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="PASS",
                            details=(
                                "chunks.embedding and skill_chunks.embedding dimensions match "
                                f"settings ({chunks_dim})."
                            ),
                        )
                    )
        except Exception as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="PostgreSQL connectivity",
                    status="FAIL",
                    details=str(exc),
                    remediation="Ensure PostgreSQL is running and PG_DSN points to a reachable instance.",
                )
                )
            checks.append(
                DoctorCheckResult(
                    name="Embedding schema alignment",
                    status="SKIP",
                    details="Skipped because database connectivity failed.",
                )
            )
    else:
        checks.append(
            DoctorCheckResult(
                name="PostgreSQL connectivity",
                status="SKIP",
                details="Skipped by --skip-db.",
            )
        )
        checks.append(
            DoctorCheckResult(
                name="Embedding schema alignment",
                status="SKIP",
                details="Skipped by --skip-db.",
            )
        )

    if not needs_ollama:
        checks.append(
            DoctorCheckResult(
                name="Ollama API reachability",
                status="SKIP",
                details="No Ollama provider selected.",
            )
        )
    elif not check_ollama:
        checks.append(
            DoctorCheckResult(
                name="Ollama API reachability",
                status="SKIP",
                details="Skipped by --skip-ollama.",
            )
        )
    else:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        required_models = _selected_ollama_models(settings)
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=timeout_seconds) as resp:
                code = int(resp.getcode() or 0)
                body = resp.read().decode("utf-8")
            if 200 <= code < 300:
                payload = json.loads(body or "{}")
                available_models = {
                    str(item.get("name")).strip()
                    for item in payload.get("models", [])
                    if isinstance(item, dict) and item.get("name")
                }
                missing_models = sorted(set(required_models.values()) - available_models)
                if missing_models:
                    checks.append(
                        DoctorCheckResult(
                            name="Ollama API reachability",
                            status="FAIL",
                            details=(
                                f"HTTP {code} from {url}, but missing configured Ollama model(s): "
                                f"{', '.join(missing_models)}."
                            ),
                            remediation=(
                                "Pull or create the missing Ollama models, or update the "
                                "OLLAMA_*_MODEL settings to models available at OLLAMA_BASE_URL."
                            ),
                        )
                    )
                else:
                    checks.append(
                        DoctorCheckResult(
                            name="Ollama API reachability",
                            status="PASS",
                            details=(
                                f"HTTP {code} from {url}. Available configured models: "
                                f"{', '.join(sorted(set(required_models.values())))}."
                            ),
                        )
                    )
            else:
                checks.append(
                    DoctorCheckResult(
                        name="Ollama API reachability",
                        status="FAIL",
                        details=f"Received HTTP {code} from {url}.",
                        remediation="Ensure Ollama is running and OLLAMA_BASE_URL is correct.",
                    )
                )
        except json.JSONDecodeError as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=f"Invalid JSON from {url}: {exc}",
                    remediation="Ensure OLLAMA_BASE_URL points to a working Ollama API endpoint.",
                )
            )
        except URLError as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=str(exc.reason),
                    remediation="Start Ollama or update OLLAMA_BASE_URL to a reachable endpoint.",
                )
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=str(exc),
                    remediation="Start Ollama or update OLLAMA_BASE_URL to a reachable endpoint.",
                )
            )

    table = Table(title="Doctor Preflight Results")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    for row in checks:
        status_style = _doctor_status_style(row.status)
        table.add_row(row.name, f"[{status_style}]{row.status}[/{status_style}]", row.details)
    console.print(table)

    remediations = [row.remediation for row in checks if row.remediation]
    if remediations:
        unique_remediations: List[str] = []
        for remediation in remediations:
            if remediation not in unique_remediations:
                unique_remediations.append(remediation)
        console.print(Panel("\n".join(f"- {item}" for item in unique_remediations), title="Suggested Fixes"))

    fail_count = sum(1 for row in checks if row.status == "FAIL")
    warn_count = sum(1 for row in checks if row.status == "WARN")

    if fail_count > 0 or (strict and warn_count > 0):
        raise typer.Exit(code=1)

    console.print("[green]Doctor checks passed.[/green]")


@app.command()
def demo(
    scenario: str = typer.Option("all", "--scenario", "-s", help="Scenario name, or 'all'."),
    list_scenarios: bool = typer.Option(False, "--list-scenarios", help="List available demo scenarios and exit."),
    max_turns: int = typer.Option(0, "--max-turns", help="Max prompts per scenario (0 = all)."),
    force_agent: bool = typer.Option(False, "--force-agent", help="Force AGENT path for every demo prompt."),
    session_mode: str = typer.Option(
        "scenario",
        "--session-mode",
        help="Session isolation mode: scenario (new session per scenario) or suite (one shared session).",
    ),
    verify: bool = typer.Option(False, "--verify", help="Run heuristic response checks and print PASS/WARN/FAIL."),
    show_notes: bool = typer.Option(False, "--show-notes", help="Show scenario briefing notes before execution."),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a curated demo suite across multiple capabilities."""

    settings = load_settings(dotenv)
    scenarios = load_demo_scenarios(settings.data_dir)
    if list_scenarios:
        console.print("[bold]Available scenarios:[/bold]")
        for scenario_obj in scenarios.values():
            console.print(render_scenario_summary(scenario_obj))
        return

    session_mode = session_mode.strip().lower()
    if session_mode not in {"scenario", "suite"}:
        raise typer.BadParameter("--session-mode must be one of: scenario, suite")

    selected_names = list(scenarios.keys()) if scenario == "all" else [scenario]
    missing = [name for name in selected_names if name not in scenarios]
    if missing:
        raise typer.BadParameter(
            f"Unknown scenario(s): {', '.join(missing)}. "
            f"Use --list-scenarios to see valid names."
        )

    demo_settings = _with_demo_settings(settings)
    if demo_settings.ollama_num_predict != settings.ollama_num_predict:
        console.print(
            f"[cyan]Demo mode override:[/cyan] OLLAMA_NUM_PREDICT={demo_settings.ollama_num_predict}"
        )

    bot = _build_bot_or_exit(demo_settings)
    shared_session = _make_local_session(dotenv, conversation_id="demo-suite") if session_mode == "suite" else None
    if shared_session is not None:
        shared_session.demo_mode = True
    if shared_session is not None and upload:
        console.print("[bold]Ingesting demo uploads for suite session...[/bold]")
        bot.ingest_and_summarize_uploads(shared_session, upload)

    verify_pass = 0
    verify_warn = 0
    verify_fail = 0

    for name in selected_names:
        scenario_obj = scenarios[name]
        turns = list(scenario_obj.turns)
        if max_turns > 0:
            turns = turns[:max_turns]

        if shared_session is None:
            session = _make_local_session(dotenv, conversation_id=f"demo-{name}")
            session.demo_mode = True
            if upload:
                console.print(f"[bold]Ingesting demo uploads for scenario '{name}'...[/bold]")
                bot.ingest_and_summarize_uploads(session, upload)
        else:
            session = shared_session

        header = f"{scenario_obj.id} - {scenario_obj.title} ({len(turns)} turn(s), difficulty={scenario_obj.difficulty})"
        console.print(Panel(header, title="Scenario"))
        console.print(f"[bold]Goal:[/bold] {scenario_obj.goal}")
        if scenario_obj.tool_focus:
            console.print(f"[bold]Tool focus:[/bold] {', '.join(scenario_obj.tool_focus)}")
        if show_notes and scenario_obj.notes:
            console.print(Panel(_render_demo_notes(scenario_obj), title="Scenario Notes"))

        for i, turn in enumerate(turns, start=1):
            console.print(f"[bold cyan]You[{i}]>[/bold cyan] {turn.prompt}")
            try:
                response = bot.process_turn(
                    session,
                    user_text=turn.prompt,
                    upload_paths=[],
                    force_agent=_coerce_force_agent(force_agent, turn),
                )
                console.print(Panel(response, title="Assistant"))

                if verify:
                    result = evaluate_response(
                        response,
                        scenario=scenario_obj,
                        turn=turn,
                    )
                    style = _verify_status_style(result.status)
                    console.print(f"[{style}]VERIFY {result.status}[/{style}] [{name} turn {i}]")
                    for message in result.messages:
                        console.print(f"- {message}")

                    if result.status == "PASS":
                        verify_pass += 1
                    elif result.status == "WARN":
                        verify_warn += 1
                    else:
                        verify_fail += 1
                        if not continue_on_error:
                            raise typer.Exit(code=1)
            except Exception as e:
                console.print(Panel(f"Demo prompt failed: {e}", title="Error"))
                if verify:
                    verify_fail += 1
                if not continue_on_error:
                    raise typer.Exit(code=1)

    if verify:
        summary = (
            f"PASS={verify_pass}  "
            f"WARN={verify_warn}  "
            f"FAIL={verify_fail}"
        )
        console.print(Panel(summary, title="Verification Summary"))


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Run the OpenAI-compatible FastAPI gateway."""
    import uvicorn

    uvicorn.run("agentic_chatbot.api.main:app", host=host, port=port, reload=reload, factory=False)
