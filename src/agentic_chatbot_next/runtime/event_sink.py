from __future__ import annotations

from agentic_chatbot_next.observability.events import RuntimeEvent


class RuntimeEventSink:
    def emit(self, event: RuntimeEvent) -> None:
        raise NotImplementedError


class NullEventSink(RuntimeEventSink):
    def emit(self, event: RuntimeEvent) -> None:
        return None
