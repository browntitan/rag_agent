from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from agentic_chatbot.runtime.context import utc_now_iso


@dataclass
class RuntimeEvent:
    event_type: str
    session_id: str
    created_at: str = field(default_factory=utc_now_iso)
    job_id: str = ""
    agent_name: str = ""
    tool_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeEventSink:
    def emit(self, event: RuntimeEvent) -> None:
        raise NotImplementedError


class NullRuntimeEventSink(RuntimeEventSink):
    def emit(self, event: RuntimeEvent) -> None:
        return None
