from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic_chatbot.config import Settings


@dataclass(frozen=True)
class RequestContext:
    """Request identity and scope for the next runtime."""

    tenant_id: str
    user_id: str
    conversation_id: str
    request_id: str = ""

    @property
    def session_id(self) -> str:
        return f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"


def build_local_context(
    settings: Settings,
    *,
    conversation_id: Optional[str] = None,
    request_id: str = "",
) -> RequestContext:
    return RequestContext(
        tenant_id=settings.default_tenant_id,
        user_id=settings.default_user_id,
        conversation_id=conversation_id or settings.default_conversation_id,
        request_id=request_id,
    )
