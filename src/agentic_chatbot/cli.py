from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from agentic_chatbot.config import load_settings
from agentic_chatbot.providers import build_providers
from agentic_chatbot.agents.orchestrator import ChatbotApp
from agentic_chatbot.agents.session import ChatSession


app = typer.Typer(add_completion=False)
console = Console()

_DEFAULT_DEMO_SCENARIOS: Dict[str, List[str]] = {
    "router_and_basic": [
        "What is fan-out vs fan-in in agentic systems?",
        "Give a concise explanation of retrieval-augmented generation.",
    ],
    "utility_and_memory": [
        "What is 18% of 2400?",
        "Remember that preferred_jurisdiction is England and Wales.",
        "What value is saved under preferred_jurisdiction?",
    ],
    "kb_grounded_qa": [
        "According to runbook_incident_response.md, what is the update cadence during SEV-1 incidents? Cite sources.",
        "From api_rate_limits.md, what are the API rate limits? Include citations.",
    ],
    "cross_document_comparison": [
        "Compare api_auth.md and api_examples.md and highlight practical auth-flow differences with citations.",
        "Compare runbook_incident_response.md and runbook_oncall_handover.md: where do responsibilities overlap?",
    ],
    "requirements_and_extraction": [
        "In 04_integrations_and_tools.md, extract requirement-like statements and list them with citations.",
        "List constraints and limits mentioned in 02_pricing_and_plans.md with citations.",
    ],
}


def _make_app(dotenv: Optional[str] = None) -> ChatbotApp:
    settings = load_settings(dotenv)
    providers = build_providers(settings)
    return ChatbotApp.create(settings, providers)


def _load_demo_scenarios(data_dir: Path) -> Dict[str, List[str]]:
    path = data_dir / "demo" / "demo_scenarios.json"
    if not path.exists():
        return _DEFAULT_DEMO_SCENARIOS
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _DEFAULT_DEMO_SCENARIOS
    if not isinstance(raw, dict):
        return _DEFAULT_DEMO_SCENARIOS
    out: Dict[str, List[str]] = {}
    for name, prompts in raw.items():
        if not isinstance(name, str):
            continue
        if not isinstance(prompts, list):
            continue
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if cleaned:
            out[name] = cleaned
    return out or _DEFAULT_DEMO_SCENARIOS


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    force_agent: bool = typer.Option(False, "--force-agent"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a single-turn query."""

    bot = _make_app(dotenv)
    session = ChatSession()

    response = bot.process_turn(session, user_text=question, upload_paths=upload, force_agent=force_agent)

    console.print(Panel(response, title="Assistant"))


@app.command()
def chat(
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Start an interactive chat session. Use /upload PATH to ingest docs mid-chat."""

    bot = _make_app(dotenv)
    session = ChatSession()

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
    records = bot.ctx.stores.doc_store.list_documents()
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
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a curated demo suite across multiple capabilities."""

    settings = load_settings(dotenv)
    scenarios = _load_demo_scenarios(settings.data_dir)
    if list_scenarios:
        console.print("Available scenarios:")
        for name, prompts in scenarios.items():
            console.print(f"- {name} ({len(prompts)} prompts)")
        return

    selected_names = list(scenarios.keys()) if scenario == "all" else [scenario]
    missing = [name for name in selected_names if name not in scenarios]
    if missing:
        raise typer.BadParameter(
            f"Unknown scenario(s): {', '.join(missing)}. "
            f"Use --list-scenarios to see valid names."
        )

    providers = build_providers(settings)
    bot = ChatbotApp.create(settings, providers)
    session = ChatSession()

    if upload:
        console.print("[bold]Ingesting demo uploads...[/bold]")
        bot.ingest_and_summarize_uploads(session, upload)

    for name in selected_names:
        prompts = scenarios[name]
        if max_turns > 0:
            prompts = prompts[:max_turns]
        console.print(Panel(f"{name} ({len(prompts)} prompt(s))", title="Scenario"))
        for i, prompt in enumerate(prompts, start=1):
            console.print(f"[bold cyan]You[{i}]>[/bold cyan] {prompt}")
            try:
                response = bot.process_turn(
                    session,
                    user_text=prompt,
                    upload_paths=[],
                    force_agent=force_agent,
                )
                console.print(Panel(response, title="Assistant"))
            except Exception as e:
                console.print(Panel(f"Demo prompt failed: {e}", title="Error"))
                if not continue_on_error:
                    raise typer.Exit(code=1)
