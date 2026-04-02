from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from agentic_chatbot.config import Settings
from agentic_chatbot.providers import ProviderBundle
from agentic_chatbot_next.app.service import RuntimeService
from agentic_chatbot_next.rag import (
    KnowledgeStores,
    SkillIndexSync,
    ensure_kb_indexed,
    ingest_paths,
    load_basic_chat_skills,
    load_stores,
)
from agentic_chatbot_next.runtime.kernel import RuntimeKernel as HybridRuntimeKernel

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings
    providers: ProviderBundle
    stores: KnowledgeStores


class ChatbotApp:
    """Deprecated compatibility wrapper around RuntimeService.

    The live runtime now lives under ``agentic_chatbot_next``. This class is
    kept as a thin delegate for one release so in-process callers do not break
    abruptly.
    """

    def __init__(self, ctx: AppContext) -> None:
        warnings.warn(
            "ChatbotApp is deprecated; use agentic_chatbot_next.app.service.RuntimeService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ctx = ctx
        self._service = RuntimeService(ctx)
        self._runtime = self._service.kernel

    @classmethod
    def create(cls, settings: Settings, providers: ProviderBundle) -> "ChatbotApp":
        stores = load_stores(settings, providers.embeddings)
        return cls(AppContext(settings=settings, providers=providers, stores=stores))

    def ingest_and_summarize_uploads(self, session: Any, upload_paths: list[Any]):
        return self._service.ingest_and_summarize_uploads(session, upload_paths)

    def process_turn(
        self,
        session: Any,
        *,
        user_text: str,
        upload_paths=None,
        force_agent: bool = False,
    ) -> str:
        return self._service.process_turn(
            session,
            user_text=user_text,
            upload_paths=upload_paths,
            force_agent=force_agent,
        )
