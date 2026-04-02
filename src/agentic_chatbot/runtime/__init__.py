from agentic_chatbot.runtime.agent_registry import RuntimeAgentRegistry
from agentic_chatbot.runtime.context import (
    AgentDefinition,
    JobRecord,
    RuntimeMessage,
    SessionState,
    TaskNotification,
    ToolContext,
    ToolSpec,
    WorkerMailboxMessage,
)

__all__ = [
    "AgentDefinition",
    "HybridRuntimeKernel",
    "JobRecord",
    "RuntimeAgentRegistry",
    "RuntimeMessage",
    "SessionState",
    "TaskNotification",
    "ToolContext",
    "ToolSpec",
    "WorkerMailboxMessage",
]


def __getattr__(name: str):
    if name == "HybridRuntimeKernel":
        from agentic_chatbot.runtime.kernel import HybridRuntimeKernel  # noqa: PLC0415

        return HybridRuntimeKernel
    raise AttributeError(name)
