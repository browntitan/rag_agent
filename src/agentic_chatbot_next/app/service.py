from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage

from agentic_chatbot_next.context import build_local_context
from agentic_chatbot_next.rag import (
    KnowledgeStores,
    SkillIndexSync,
    ensure_kb_indexed,
    ingest_paths,
    load_basic_chat_skills,
    load_stores,
)
from agentic_chatbot_next.providers.factory import ProviderBundle
from agentic_chatbot_next.rag.engine import render_rag_contract, run_rag_contract
from agentic_chatbot_next.router.llm_router import route_turn
from agentic_chatbot_next.router.policy import choose_agent_name
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

logger = logging.getLogger(__name__)


def _summarise_history(messages: List[Any], n: int = 2) -> str:
    human_ai_pairs: List[str] = []
    for msg in reversed(messages):
        role = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "") or "").strip()
        if role in ("human", "ai") and content:
            prefix = "User" if role == "human" else "Assistant"
            human_ai_pairs.append(f"{prefix}: {content[:120]}")
        if len(human_ai_pairs) >= n * 2:
            break
    return "\n".join(reversed(human_ai_pairs))


@dataclass
class AppContext:
    settings: Any
    providers: ProviderBundle
    stores: KnowledgeStores


class RuntimeService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._basic_chat_system_prompt = load_basic_chat_skills(ctx.settings)
        self.kernel = RuntimeKernel(ctx.settings, providers=ctx.providers, stores=ctx.stores)
        if getattr(ctx.settings, "agent_runtime_mode", ""):
            logger.info(
                "AGENT_RUNTIME_MODE=%s is deprecated and ignored by agentic_chatbot_next.",
                ctx.settings.agent_runtime_mode,
            )
        if getattr(ctx.settings, "agent_definitions_json", ""):
            logger.info("AGENT_DEFINITIONS_JSON is deprecated and ignored by agentic_chatbot_next.")
        try:
            SkillIndexSync(self.ctx.settings, self.ctx.stores).sync(
                tenant_id=self.ctx.settings.default_tenant_id,
            )
        except Exception as exc:
            logger.warning("Could not sync skill packs at startup: %s", exc)
        self._ensure_kb_ready(self.ctx.settings.default_tenant_id)

    @classmethod
    def create(cls, settings: Any, providers: ProviderBundle) -> "RuntimeService":
        stores = load_stores(settings, providers.embeddings)
        return cls(AppContext(settings=settings, providers=providers, stores=stores))

    def _ensure_workspace(self, session: Any) -> None:
        if getattr(session, "workspace", None) is not None:
            return
        if getattr(self.ctx.settings, "workspace_dir", None) is None:
            return
        try:
            workspace = SessionWorkspace.for_session(session.session_id, self.ctx.settings.workspace_dir)
            workspace.open()
            session.workspace = workspace
            logger.debug("Opened session workspace at %s", workspace.root)
        except Exception as exc:
            logger.warning("Could not open session workspace: %s", exc)

    def _ensure_kb_ready(self, tenant_id: str) -> None:
        if self.ctx.stores is None or not hasattr(self.ctx.stores, "doc_store"):
            return
        try:
            ensure_kb_indexed(self.ctx.settings, self.ctx.stores, tenant_id=tenant_id)
        except Exception as exc:
            logger.warning("Could not ensure KB index readiness: %s", exc)

    def ingest_and_summarize_uploads(
        self,
        session: Any,
        upload_paths: List[Path],
    ) -> Tuple[List[str], str]:
        callbacks = self.kernel.build_callbacks(
            session,
            trace_name="upload_ingest",
            agent_name="rag_worker",
            metadata={
                "num_files": len(upload_paths),
                "tenant_id": session.tenant_id,
                "user_id": session.user_id,
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
            },
        )

        doc_ids = ingest_paths(
            self.ctx.settings,
            self.ctx.stores,
            upload_paths,
            source_type="upload",
            tenant_id=session.tenant_id,
        )
        session.uploaded_doc_ids.extend([doc_id for doc_id in doc_ids if doc_id not in session.uploaded_doc_ids])

        if getattr(session, "workspace", None) is not None:
            for upload_path in upload_paths:
                try:
                    session.workspace.copy_file(upload_path)
                    logger.debug("Copied %s into session workspace", upload_path.name)
                except Exception as exc:
                    logger.warning("Could not copy %s into workspace: %s", upload_path.name, exc)

        if not doc_ids:
            return [], "No documents were ingested (files missing or already indexed)."

        summary_query = (
            "Summarize the uploaded documents. Provide:\n"
            "1) A 6-bullet executive summary\n"
            "2) Key definitions / terminology\n"
            "3) Important numbers / constraints (if any)\n"
            "4) Open questions / ambiguities\n"
            "5) 5 suggested questions the user can ask next\n"
            "Cite evidence inline using (citation_id)."
        )
        rag_out = self._call_rag_direct(
            session=session,
            query=summary_query,
            conversation_context="User uploaded documents.",
            preferred_doc_ids=doc_ids,
            callbacks=callbacks,
        )
        rendered = render_rag_contract(rag_out)
        session.messages.append(AIMessage(content=rendered))
        return doc_ids, rendered

    def _call_rag_direct(
        self,
        *,
        session: Any,
        query: str,
        conversation_context: str,
        preferred_doc_ids: List[str],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        contract = run_rag_contract(
            self.ctx.settings,
            self.ctx.stores,
            providers=self.ctx.providers,
            session=session,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=True,
            top_k_vector=self.ctx.settings.rag_top_k_vector,
            top_k_keyword=self.ctx.settings.rag_top_k_keyword,
            max_retries=self.ctx.settings.rag_max_retries,
            callbacks=callbacks or [],
        )
        return contract.to_dict()

    def process_turn(
        self,
        session: Any,
        *,
        user_text: str,
        upload_paths: Optional[List[Path]] = None,
        force_agent: bool = False,
    ) -> str:
        upload_paths = upload_paths or []
        self._ensure_workspace(session)
        self._ensure_kb_ready(session.tenant_id)

        if upload_paths:
            self.ingest_and_summarize_uploads(session, upload_paths)

        decision = route_turn(
            self.ctx.settings,
            self.ctx.providers,
            user_text=user_text,
            has_attachments=bool(upload_paths),
            history_summary=_summarise_history(session.messages, n=2),
            force_agent=force_agent,
        )

        meta = {
            "route": decision.route,
            "router_confidence": decision.confidence,
            "router_reasons": decision.reasons,
            "router_method": getattr(decision, "router_method", "deterministic"),
            "suggested_agent": getattr(decision, "suggested_agent", ""),
            "has_attachments": bool(upload_paths),
            "uploaded_doc_ids": list(getattr(session, "uploaded_doc_ids", []) or []),
            "tenant_id": session.tenant_id,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "request_id": session.request_id,
        }
        self.kernel.emit_router_decision(
            session,
            route=decision.route,
            confidence=decision.confidence,
            reasons=list(decision.reasons),
            router_method=getattr(decision, "router_method", "deterministic"),
            suggested_agent=getattr(decision, "suggested_agent", ""),
            force_agent=force_agent,
            has_attachments=bool(upload_paths),
        )

        if decision.route == "BASIC":
            text = self.kernel.process_basic_turn(
                session,
                user_text=user_text,
                system_prompt=self._basic_chat_system_prompt,
                chat_llm=self.ctx.providers.chat,
                route_metadata=meta,
            )
            if getattr(self.ctx.settings, "clear_scratchpad_per_turn", False):
                session.clear_scratchpad()
            return text

        requested_agent = choose_agent_name(self.ctx.settings, decision) or "general"
        try:
            text = self.kernel.process_agent_turn(
                session,
                user_text=user_text,
                agent_name=requested_agent,
                route_metadata=meta,
            )
        finally:
            if getattr(self.ctx.settings, "clear_scratchpad_per_turn", False):
                session.clear_scratchpad()
        return text

    @classmethod
    def from_settings(cls, settings: Any, providers: Optional[ProviderBundle] = None) -> "RuntimeService":
        from agentic_chatbot_next.providers.factory import build_providers

        resolved_providers = providers or build_providers(settings)
        return cls.create(settings, resolved_providers)

    @classmethod
    def create_local_session(cls, settings: Any, *, conversation_id: Optional[str] = None) -> Any:
        from agentic_chatbot_next.session import ChatSession

        ctx = build_local_context(settings, conversation_id=conversation_id)
        return ChatSession.from_context(ctx)
