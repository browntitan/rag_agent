from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_chatbot.config import Settings


def get_langchain_callbacks(
    settings: Settings,
    *,
    session_id: str,
    trace_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Return LangChain callback handlers.

    If Langfuse env vars are not set, returns an empty list.

    The returned objects are used as `config={"callbacks": callbacks}` in
    LangChain/LangGraph invocations.
    """

    metadata = metadata or {}

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []

    try:
        from langfuse.callback import CallbackHandler

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                debug=settings.langfuse_debug,
                session_id=session_id,
                trace_name=trace_name,
                metadata=metadata,
            )
        ]
    except Exception:
        # If Langfuse isn't installed or the API changed, don't break the app.
        return []
