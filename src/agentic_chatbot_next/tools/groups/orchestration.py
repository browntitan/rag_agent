from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import tool


def build_orchestration_tools(ctx: Any) -> List[Any]:
    if ctx.kernel is None or ctx.active_definition is None:
        return []

    @tool
    def spawn_worker(
        prompt: str,
        agent_name: str = "utility",
        description: str = "",
        run_in_background: bool = False,
    ) -> str:
        """Spawn a scoped worker from the current next runtime."""

        return json.dumps(
            ctx.kernel.spawn_worker_from_tool(
                ctx,
                prompt=prompt,
                agent_name=agent_name,
                description=description,
                run_in_background=run_in_background,
            ),
            ensure_ascii=False,
        )

    @tool
    def message_worker(job_id: str, message: str, resume: bool = True) -> str:
        """Queue a follow-up message for an existing worker job."""

        return json.dumps(
            ctx.kernel.message_worker_from_tool(
                ctx,
                job_id=job_id,
                message=message,
                resume=resume,
            ),
            ensure_ascii=False,
        )

    @tool
    def list_jobs(status_filter: str = "") -> str:
        """List durable runtime jobs for the current session."""

        return json.dumps(
            ctx.kernel.list_jobs_from_tool(
                ctx,
                status_filter=status_filter,
            ),
            ensure_ascii=False,
        )

    @tool
    def stop_job(job_id: str) -> str:
        """Stop a background worker job."""

        return json.dumps(
            ctx.kernel.stop_job_from_tool(
                ctx,
                job_id=job_id,
            ),
            ensure_ascii=False,
        )

    return [spawn_worker, message_worker, list_jobs, stop_job]
