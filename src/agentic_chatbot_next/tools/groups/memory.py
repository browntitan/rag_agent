from __future__ import annotations

from typing import Any, List

from langchain_core.tools import tool

from agentic_chatbot_next.memory.scope import MemoryScope


def build_memory_tools(ctx: Any) -> List[Any]:
    if ctx.file_memory_store is None:
        return []

    @tool
    def memory_save(key: str, value: str, scope: str = "conversation") -> str:
        """Save a fact to file-backed memory.

        Args:
            key: Short identifier for the memory entry.
            value: The value to remember.
            scope: conversation or user.
        """

        scope_value = MemoryScope(scope).value
        ctx.file_memory_store.save(
            tenant_id=ctx.session.tenant_id,
            user_id=ctx.session.user_id,
            conversation_id=ctx.session.conversation_id,
            scope=scope_value,
            key=key,
            value=value,
        )
        return f"Saved memory in {scope_value} scope: {key!r} = {value!r}"

    @tool
    def memory_load(key: str, scope: str = "conversation") -> str:
        """Load a fact from file-backed memory by key."""

        scope_value = MemoryScope(scope).value
        value = ctx.file_memory_store.get(
            tenant_id=ctx.session.tenant_id,
            user_id=ctx.session.user_id,
            conversation_id=ctx.session.conversation_id,
            scope=scope_value,
            key=key,
        )
        if value is None:
            return f"No memory found for key {key!r} in {scope_value} scope."
        return value

    @tool
    def memory_list(scope: str = "conversation") -> str:
        """List keys in file-backed memory for the selected scope."""

        scope_value = MemoryScope(scope).value
        keys = ctx.file_memory_store.list_keys(
            tenant_id=ctx.session.tenant_id,
            user_id=ctx.session.user_id,
            conversation_id=ctx.session.conversation_id,
            scope=scope_value,
        )
        if not keys:
            return f"No memory keys saved for {scope_value} scope."
        return ", ".join(keys)

    return [memory_save, memory_load, memory_list]
