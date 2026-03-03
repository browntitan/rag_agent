from __future__ import annotations

from typing import Any, List

from langchain.tools import tool


def make_memory_tools(stores: Any, session: Any) -> List[Any]:
    """Return LangChain tools for persistent cross-turn memory.

    Tools returned:
      - memory_save   — persist a key/value fact to the database
      - memory_load   — retrieve a fact by key
      - memory_list   — list all saved memory keys for this session
    """
    memory_store = stores.memory_store

    @tool
    def memory_save(key: str, value: str) -> str:
        """Save a fact to persistent memory for recall in future turns.

        Args:
          key:   Short identifier (e.g. 'user_name', 'preferred_format').
          value: The value to remember.

        Returns a confirmation string.
        """
        memory_store.save(session.session_id, key, value)
        return f"Saved memory: {key!r} = {value!r}"

    @tool
    def memory_load(key: str) -> str:
        """Load a previously saved fact from memory by key.

        Args:
          key: The key used when saving (e.g. 'user_name').

        Returns the stored value, or a message indicating it was not found.
        """
        value = memory_store.get(session.session_id, key)
        if value is None:
            return f"No memory found for key {key!r}."
        return value

    @tool
    def memory_list() -> str:
        """List all keys saved in persistent memory for this session.

        Returns a comma-separated list of keys, or a message if none exist.
        """
        keys = memory_store.list_keys(session.session_id)
        if not keys:
            return "No memory keys saved for this session."
        return ", ".join(keys)

    return [memory_save, memory_load, memory_list]
