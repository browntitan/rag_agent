"""Tool assembly helpers for the next runtime."""
from __future__ import annotations

from typing import Any, List

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.tools.policy import ToolPolicyService
from agentic_chatbot_next.tools.registry import build_tool_definitions


def build_agent_tools(
    agent: AgentDefinition,
    tool_context: Any,
    *,
    policy_service: ToolPolicyService | None = None,
) -> List[Any]:
    policy = policy_service or ToolPolicyService()
    definitions = build_tool_definitions(tool_context)
    tools: List[Any] = []
    for tool_name in agent.allowed_tools:
        definition = definitions.get(tool_name)
        if definition is None:
            continue
        if not policy.is_allowed(agent, definition, tool_context):
            continue
        for candidate in definition.builder(tool_context):
            if getattr(candidate, "name", "") == tool_name:
                tools.append(candidate)
                break
    return tools
