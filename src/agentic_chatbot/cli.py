from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from agentic_chatbot.config import load_settings
from agentic_chatbot.context import build_local_context
from agentic_chatbot.demo import (
    DemoScenario,
    DemoTurn,
    evaluate_response,
    load_demo_scenarios,
    render_scenario_summary,
)
from agentic_chatbot.providers import build_providers
from agentic_chatbot.agents.orchestrator import ChatbotApp
from agentic_chatbot.agents.session import ChatSession


app = typer.Typer(add_completion=False)
console = Console()

def _make_app(dotenv: Optional[str] = None) -> ChatbotApp:
    settings = load_settings(dotenv)
    providers = build_providers(settings)
    return ChatbotApp.create(settings, providers)


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


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    force_agent: bool = typer.Option(False, "--force-agent"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a single-turn query."""

    bot = _make_app(dotenv)
    session = _make_local_session(dotenv)

    response = bot.process_turn(session, user_text=question, upload_paths=upload, force_agent=force_agent)

    console.print(Panel(response, title="Assistant"))


@app.command()
def chat(
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Start an interactive chat session. Use /upload PATH to ingest docs mid-chat."""

    bot = _make_app(dotenv)
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
    """Force (re)indexing of the built-in demo KB."""

    bot = _make_app(dotenv)
    # `ensure_kb_indexed` already runs on init; this command just confirms.
    console.print("KB indexing ensured. Indexed documents:")
    tenant_id = bot.ctx.settings.default_tenant_id
    records = bot.ctx.stores.doc_store.list_documents(tenant_id=tenant_id)
    docs = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "source_type": r.source_type,
            "num_chunks": r.num_chunks,
            "doc_structure_type": r.doc_structure_type,
        }
        for r in records
    ]
    console.print(json.dumps(docs, indent=2, ensure_ascii=False)[:4000])


@app.command()
def reset_indexes(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Truncate all indexed data from PostgreSQL (documents, chunks, memory)."""

    settings = load_settings(dotenv)

    if not confirm:
        typer.confirm(
            "This will DELETE all documents, chunks, and memory from the database. Continue?",
            abort=True,
        )

    from agentic_chatbot.db.connection import apply_schema, get_conn, init_pool

    init_pool(settings)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()

    console.print("All indexes cleared. Run init-kb or chat to rebuild.")


@app.command()
def migrate(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Apply the database schema (idempotent — safe to run multiple times)."""

    settings = load_settings(dotenv)
    from agentic_chatbot.db.connection import apply_schema

    apply_schema(settings)
    console.print("Schema applied successfully.")


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

    providers = build_providers(settings)
    bot = ChatbotApp.create(settings, providers)
    shared_session = _make_local_session(dotenv, conversation_id="demo-suite") if session_mode == "suite" else None
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
