"""Lightweight stand-in for ChatSession used by graph nodes.

Tools created via ``make_all_rag_tools(stores, session)`` and
``make_memory_tools(stores, session)`` close over a session object.
They only access ``session.scratchpad``, ``session.session_id``,
``session.tenant_id``, ``session.uploaded_doc_ids``, and optionally
``session.workspace``.  This proxy provides those same attributes
so the existing tool factories work without modification (duck typing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agentic_chatbot.sandbox.session_workspace import SessionWorkspace


@dataclass
class SessionProxy:
    """Bridges AgentState fields to the interface tools expect."""

    session_id: str = ""
    tenant_id: str = "local-dev"
    demo_mode: bool = False
    scratchpad: Dict[str, str] = field(default_factory=dict)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    # messages is unused by tools but satisfies any hasattr checks
    messages: List[Any] = field(default_factory=list)
    # workspace is the persistent session directory shared by all agents.
    # Set by builder.py from ChatSession.workspace; None when workspace is
    # unavailable or not yet opened.
    workspace: Optional["SessionWorkspace"] = field(default=None, repr=False)

    def clear_scratchpad(self) -> None:
        self.scratchpad.clear()
