from __future__ import annotations

from typing import Any

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition


class ToolPolicyService:
    """Central tool policy for the next runtime."""

    READ_ONLY_ONLY_MODES = {"basic", "planner", "finalizer", "verifier", "rag"}

    def is_allowed(
        self,
        agent: AgentDefinition,
        tool: ToolDefinition | str,
        tool_context: Any | None = None,
    ) -> bool:
        tool_name = tool if isinstance(tool, str) else tool.name
        if tool_name not in set(agent.allowed_tools):
            return False
        if isinstance(tool, str):
            return True

        if tool.requires_workspace:
            workspace_root = getattr(tool_context, "workspace_root", None) if tool_context is not None else None
            if workspace_root is None:
                return False

        task_payload = {}
        if tool_context is not None:
            task_payload = dict((tool_context.metadata or {}).get("task_payload") or {})
        if task_payload.get("job_id") and not tool.background_safe:
            return False

        if agent.mode in self.READ_ONLY_ONLY_MODES and not tool.read_only:
            return False

        if agent.mode == "memory_maintainer" and tool.group != "memory":
            return False

        if bool((tool_context.metadata if tool_context is not None else {}).get("read_only_only")) and not tool.read_only:
            return False

        return True
