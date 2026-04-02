from __future__ import annotations

from typing import Iterable, List

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.scope import MemoryScope


class MemoryContextBuilder:
    def __init__(self, store: FileMemoryStore, *, max_chars: int = 2000) -> None:
        self.store = store
        self.max_chars = max_chars

    def build_for_agent(self, agent: AgentDefinition, session_state: SessionState) -> str:
        scopes = self._normalise_scopes(agent.memory_scopes)
        parts: List[str] = []
        consumed = 0
        for scope in scopes:
            entries = self.store.list_entries(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                scope=scope,
            )
            if not entries:
                continue
            lines = [f"[{scope} memory]"]
            for entry in entries:
                line = f"- {entry.key}: {entry.value}"
                if consumed + len(line) > self.max_chars and parts:
                    break
                lines.append(line)
                consumed += len(line) + 1
            block = "\n".join(lines)
            if len(lines) > 1 and block.strip():
                parts.append(block)
            if consumed >= self.max_chars:
                break
        return "\n\n".join(parts).strip()

    def _normalise_scopes(self, scopes: Iterable[str]) -> List[str]:
        clean = []
        for scope in scopes:
            try:
                clean.append(MemoryScope(scope).value)
            except ValueError:
                continue
        return clean or [MemoryScope.conversation.value]
