from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from agentic_chatbot.config import Settings


@dataclass(frozen=True)
class RequestContext:
    """Request identity and scope used for tenant isolation and memory keys."""

    tenant_id: str
    user_id: str
    conversation_id: str
    request_id: str = ""

    @property
    def session_id(self) -> str:
        """Stable session key shared by memory/tools within this conversation."""
        return f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"


def build_local_context(
    settings: Settings,
    *,
    conversation_id: Optional[str] = None,
    request_id: str = "",
) -> RequestContext:
    """Build default local/dev context for CLI and demos."""
    return RequestContext(
        tenant_id=settings.default_tenant_id,
        user_id=settings.default_user_id,
        conversation_id=conversation_id or settings.default_conversation_id,
        request_id=request_id,
    )


def build_context_from_claims(
    settings: Settings,
    claims: Mapping[str, object],
    *,
    conversation_id: Optional[str] = None,
    request_id: str = "",
    fallback_user_id: Optional[str] = None,
) -> RequestContext:
    """Create RequestContext from JWT claims and request metadata."""
    tenant_id = str(claims.get("tenant_id") or "").strip()
    user_id = str(claims.get("sub") or fallback_user_id or "").strip()

    if not tenant_id:
        raise ValueError("JWT is missing required claim: tenant_id")
    if not user_id:
        raise ValueError("JWT is missing required claim: sub")

    return RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=(conversation_id or "default").strip() or "default",
        request_id=request_id,
    )
