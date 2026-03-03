from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agentic_chatbot.context import RequestContext


@dataclass
class ChatSession:
    """Conversation state for the demo chatbot."""

    tenant_id: str = "local-dev"
    user_id: str = "local-cli"
    conversation_id: str = "local-session"
    request_id: str = ""
    session_id: str = ""
    messages: List[Any] = field(default_factory=list)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"

    @classmethod
    def from_context(cls, ctx: RequestContext, *, messages: List[Any] | None = None) -> "ChatSession":
        return cls(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            messages=list(messages or []),
        )

    def clear_scratchpad(self) -> None:
        """Clear all within-turn working memory."""
        self.scratchpad.clear()
