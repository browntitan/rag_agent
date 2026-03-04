from __future__ import annotations

import inspect
import os
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
        # Langfuse import path changed across versions.
        try:
            from langfuse.langchain import CallbackHandler
        except Exception:
            from langfuse.callback import CallbackHandler

        params = inspect.signature(CallbackHandler.__init__).parameters

        # Older SDKs accept explicit host/secret/session/metadata args.
        if "secret_key" in params:
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

        # Newer SDKs (v3+) read auth/host from env and expose a slimmer API.
        if settings.langfuse_host:
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key or ""
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key or ""
        try:
            # v3 callback handlers resolve clients from the global client registry.
            # Explicitly initialize the client so callbacks can attach and emit traces.
            from langfuse import Langfuse

            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                debug=settings.langfuse_debug,
            )
        except Exception:
            pass

        try:
            handler = CallbackHandler(public_key=settings.langfuse_public_key, update_trace=True)
        except TypeError:
            handler = CallbackHandler(public_key=settings.langfuse_public_key)

        return [handler]
    except Exception:
        # If Langfuse isn't installed or the API changed, don't break the app.
        return []
