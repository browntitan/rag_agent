from __future__ import annotations

from typing import Any, Dict

from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.groups.analyst import build_analyst_tools
from agentic_chatbot_next.tools.groups.memory import build_memory_tools
from agentic_chatbot_next.tools.groups.orchestration import build_orchestration_tools
from agentic_chatbot_next.tools.groups.rag_gateway import build_rag_gateway_tools
from agentic_chatbot_next.tools.groups.utility import build_utility_tools


def build_tool_definitions(ctx: Any) -> Dict[str, ToolDefinition]:
    return {
        "calculator": ToolDefinition(name="calculator", group="utility", builder=build_utility_tools, read_only=True, background_safe=True),
        "list_indexed_docs": ToolDefinition(name="list_indexed_docs", group="utility", builder=build_utility_tools, read_only=True, background_safe=True),
        "search_skills": ToolDefinition(name="search_skills", group="utility", builder=build_utility_tools, read_only=True, background_safe=True),
        "memory_save": ToolDefinition(name="memory_save", group="memory", builder=build_memory_tools, background_safe=True),
        "memory_load": ToolDefinition(name="memory_load", group="memory", builder=build_memory_tools, read_only=True, background_safe=True),
        "memory_list": ToolDefinition(name="memory_list", group="memory", builder=build_memory_tools, read_only=True, background_safe=True),
        "rag_agent_tool": ToolDefinition(name="rag_agent_tool", group="rag_gateway", builder=build_rag_gateway_tools, background_safe=True),
        "load_dataset": ToolDefinition(name="load_dataset", group="analyst", builder=build_analyst_tools, read_only=True, requires_workspace=True, background_safe=True),
        "inspect_columns": ToolDefinition(name="inspect_columns", group="analyst", builder=build_analyst_tools, read_only=True, requires_workspace=True, background_safe=True),
        "execute_code": ToolDefinition(name="execute_code", group="analyst", builder=build_analyst_tools, destructive=True, requires_workspace=True, background_safe=True),
        "scratchpad_write": ToolDefinition(name="scratchpad_write", group="analyst", builder=build_analyst_tools, background_safe=True),
        "scratchpad_read": ToolDefinition(name="scratchpad_read", group="analyst", builder=build_analyst_tools, read_only=True, background_safe=True),
        "scratchpad_list": ToolDefinition(name="scratchpad_list", group="analyst", builder=build_analyst_tools, read_only=True, background_safe=True),
        "workspace_write": ToolDefinition(name="workspace_write", group="analyst", builder=build_analyst_tools, requires_workspace=True, background_safe=True),
        "workspace_read": ToolDefinition(name="workspace_read", group="analyst", builder=build_analyst_tools, read_only=True, requires_workspace=True, background_safe=True),
        "workspace_list": ToolDefinition(name="workspace_list", group="analyst", builder=build_analyst_tools, read_only=True, requires_workspace=True, background_safe=True),
        "spawn_worker": ToolDefinition(name="spawn_worker", group="orchestration", builder=build_orchestration_tools),
        "message_worker": ToolDefinition(name="message_worker", group="orchestration", builder=build_orchestration_tools),
        "list_jobs": ToolDefinition(name="list_jobs", group="orchestration", builder=build_orchestration_tools, read_only=True),
        "stop_job": ToolDefinition(name="stop_job", group="orchestration", builder=build_orchestration_tools),
    }
