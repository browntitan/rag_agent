from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from new_demo_notebook.lib.scenario_runner import ScenarioDefinition, ScenarioResult
from new_demo_notebook.lib.trace_reader import TraceBundle


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)


def _display_pretty(value: Any) -> None:
    try:
        from IPython.display import JSON, display

        display(JSON(value))
    except Exception:
        print(json.dumps(value, indent=2, ensure_ascii=False))


def build_event_rows(bundle: TraceBundle, trace_focus: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = list(bundle.event_rows + bundle.job_events)
    if not trace_focus:
        return rows
    focus = {item.lower() for item in trace_focus}
    filtered = []
    for row in rows:
        agent_name = str(row.get("agent_name") or "").lower()
        tool_name = str(row.get("tool_name") or "").lower()
        event_type = str(row.get("event_type") or "").lower()
        if agent_name in focus or tool_name in focus or event_type in focus:
            filtered.append(row)
    return filtered


def build_job_rows(bundle: TraceBundle) -> List[Dict[str, Any]]:
    return [
        {
            "job_id": str(job.get("job_id") or ""),
            "agent_name": str(job.get("agent_name") or ""),
            "status": str(job.get("status") or ""),
            "description": str(job.get("description") or ""),
            "result_summary": str(job.get("result_summary") or ""),
        }
        for job in bundle.jobs
    ]


def display_trace_bundle(bundle: TraceBundle, *, trace_focus: Optional[List[str]] = None) -> None:
    _display_markdown(f"### Trace Summary: `{bundle.conversation_id}`")
    _display_pretty(
        {
            "conversation_id": bundle.conversation_id,
            "session_ids": bundle.session_ids,
            "event_count": len(bundle.event_rows) + len(bundle.job_events),
            "job_count": len(bundle.jobs),
            "notification_count": len(bundle.notifications),
            "workspace_files": bundle.workspace_files,
        }
    )
    _display_markdown("#### Event Timeline")
    _display_pretty(build_event_rows(bundle, trace_focus=trace_focus))
    _display_markdown("#### Jobs")
    _display_pretty(build_job_rows(bundle))
    _display_markdown("#### Transcript Excerpts")
    _display_pretty(bundle.transcript_rows[-6:])


def display_scenario_result(result: ScenarioResult) -> None:
    latest = result.attempts[-1]
    _display_markdown(
        f"## {result.scenario.title}\n"
        f"- success: `{latest.success}`\n"
        f"- observed route: `{latest.observed_route}`\n"
        f"- observed agents: `{', '.join(latest.observed_agents)}`"
    )
    if latest.validation_errors:
        _display_markdown("### Validation Errors")
        _display_pretty(latest.validation_errors)
    _display_markdown("### Assistant Outputs")
    _display_pretty(latest.outputs)
    display_trace_bundle(result.bundle, trace_focus=result.scenario.trace_focus)


def display_coverage_matrix(scenarios: Iterable[ScenarioDefinition]) -> None:
    rows = [
        {
            "scenario_id": scenario.id,
            "title": scenario.title,
            "expected_agents": scenario.expected_agents,
            "expected_route": scenario.expected_route,
        }
        for scenario in scenarios
    ]
    _display_markdown("## Agent Coverage Matrix")
    _display_pretty(rows)
