from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.observability import get_langchain_callbacks
from agentic_chatbot.providers import ProviderBundle
from agentic_chatbot.rag import (
    KnowledgeStores,
    load_basic_chat_skills,
    ensure_kb_indexed,
    ingest_paths,
    load_general_agent_skills,
    load_stores,
)
from agentic_chatbot.router import route_message, route_message_hybrid
from agentic_chatbot.tools import calculator, make_list_docs_tool, make_memory_tools, make_rag_agent_tool

from agentic_chatbot.agents.basic_chat import run_basic_chat
from agentic_chatbot.agents.general_agent import run_general_agent
from agentic_chatbot.agents.session import ChatSession

logger = logging.getLogger(__name__)


def _summarise_history(messages: List[Any], n: int = 2) -> str:
    """Return a compact plain-text summary of the last *n* message pairs.

    Used to give the LLM router context about the ongoing conversation without
    sending the full message history.
    """
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


def _is_graph_capability_error(exc: Exception) -> bool:
    """Return True when falling back to the legacy agent is expected/safe."""
    text = str(exc).lower()
    capability_markers = (
        "bind_tools",
        "tool calling",
        "tool-calling",
        "does not support tool",
        "not implemented",
        "unsupported",
    )
    if any(m in text for m in capability_markers):
        return True
    return isinstance(exc, (NotImplementedError, ModuleNotFoundError, ImportError))


@dataclass
class AppContext:
    settings: Settings
    providers: ProviderBundle
    stores: KnowledgeStores


class ChatbotApp:
    """Main orchestrator: upload ingestion -> router -> basic vs agent."""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        # Load system prompts from skills files once at startup.
        self._general_agent_system_prompt = load_general_agent_skills(ctx.settings)
        self._basic_chat_system_prompt = load_basic_chat_skills(ctx.settings)
        # Ensure KB is indexed once at startup.
        ensure_kb_indexed(
            self.ctx.settings,
            self.ctx.stores,
            tenant_id=self.ctx.settings.default_tenant_id,
        )

    @classmethod
    def create(cls, settings: Settings, providers: ProviderBundle) -> "ChatbotApp":
        stores = load_stores(settings, providers.embeddings)
        ctx = AppContext(settings=settings, providers=providers, stores=stores)
        return cls(ctx)

    def _build_tools(self, session: ChatSession) -> List[Any]:
        s = self.ctx.settings
        stores = self.ctx.stores

        rag_tool = make_rag_agent_tool(
            s,
            stores,
            llm=self.ctx.providers.chat,
            judge_llm=self.ctx.providers.judge,
            session=session,
        )
        list_docs_tool = make_list_docs_tool(s, stores, session)
        memory_tools = make_memory_tools(stores, session)

        return [calculator, rag_tool, list_docs_tool] + memory_tools

    def ingest_and_summarize_uploads(
        self,
        session: ChatSession,
        upload_paths: List[Path],
    ) -> Tuple[List[str], str]:
        """Ingest uploaded docs and return (doc_ids, summary_text)."""

        s = self.ctx.settings
        callbacks = get_langchain_callbacks(
            s,
            session_id=session.session_id,
            trace_name="upload_ingest",
            metadata={
                "num_files": len(upload_paths),
                "tenant_id": session.tenant_id,
                "user_id": session.user_id,
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
            },
        )

        doc_ids = ingest_paths(
            s,
            self.ctx.stores,
            upload_paths,
            source_type="upload",
            tenant_id=session.tenant_id,
        )
        session.uploaded_doc_ids.extend([d for d in doc_ids if d not in session.uploaded_doc_ids])

        # Copy uploaded files into the session workspace so the Docker sandbox
        # can access them immediately via the bind-mounted /workspace directory.
        if session.workspace is not None:
            for up in upload_paths:
                try:
                    session.workspace.copy_file(up)
                    logger.debug("ingest: copied upload %s into session workspace", up.name)
                except Exception as ws_exc:
                    logger.warning("ingest: could not copy %s to workspace: %s", up.name, ws_exc)

        if not doc_ids:
            return [], "No documents were ingested (files missing or already indexed)."

        # Kick off RAG tool: summarize uploaded docs with citations.
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

        rendered = self._render_rag_result(rag_out)
        # Store the summary as an assistant message in history.
        session.messages.append(AIMessage(content=rendered))

        return doc_ids, rendered

    def _call_rag_direct(
        self,
        *,
        session: ChatSession,
        query: str,
        conversation_context: str,
        preferred_doc_ids: List[str],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        callbacks = callbacks or []
        return json.loads(
            json.dumps(
                self._rag_agent(
                    session=session,
                    query=query,
                    conversation_context=conversation_context,
                    preferred_doc_ids=preferred_doc_ids,
                    callbacks=callbacks,
                ),
                ensure_ascii=False,
            )
        )

    def _rag_agent(
        self,
        *,
        session: ChatSession,
        query: str,
        conversation_context: str,
        preferred_doc_ids: List[str],
        callbacks: List[Any],
    ) -> Dict[str, Any]:
        # Direct call to agentic RAG (not as tool call) for upload kickoff.
        from agentic_chatbot.rag.agent import run_rag_agent

        return run_rag_agent(
            self.ctx.settings,
            self.ctx.stores,
            llm=self.ctx.providers.chat,
            judge_llm=self.ctx.providers.judge,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=True,
            top_k_vector=self.ctx.settings.rag_top_k_vector,
            top_k_keyword=self.ctx.settings.rag_top_k_keyword,
            max_retries=self.ctx.settings.rag_max_retries,
            session=session,
            callbacks=callbacks,
        )

    def _render_rag_result(self, rag_out: Dict[str, Any]) -> str:
        ans = rag_out.get("answer", "")
        citations = rag_out.get("citations", [])
        used = set(rag_out.get("used_citation_ids", []))

        lines = [ans.strip()]

        if citations:
            lines.append("\nCitations:")
            for c in citations:
                cid = c.get("citation_id")
                if used and cid not in used:
                    continue
                title = c.get("title", "")
                loc = c.get("location", "")
                lines.append(f"- [{cid}] {title} ({loc})")

        if rag_out.get("warnings"):
            lines.append("\nWarnings: " + ", ".join([str(w) for w in rag_out.get("warnings")]))

        if rag_out.get("followups"):
            lines.append("\nFollow-ups:")
            for q in rag_out.get("followups"):
                lines.append(f"- {q}")

        return "\n".join(lines).strip()

    def process_turn(
        self,
        session: ChatSession,
        *,
        user_text: str,
        upload_paths: Optional[List[Path]] = None,
        force_agent: bool = False,
        extra_callbacks: Optional[List[Any]] = None,
    ) -> str:
        """Process one user turn. Optionally ingest uploads first."""

        upload_paths = upload_paths or []

        # Lazily open the session workspace on the first turn so that the Docker
        # sandbox has a persistent bind-mounted directory for the entire session.
        if session.workspace is None and self.ctx.settings.workspace_dir is not None:
            try:
                from agentic_chatbot.sandbox.session_workspace import SessionWorkspace  # noqa: PLC0415
                ws = SessionWorkspace.for_session(session.session_id, self.ctx.settings.workspace_dir)
                ws.open()
                session.workspace = ws
                logger.debug("Opened session workspace at %s", ws.root)
            except Exception as ws_exc:
                logger.warning("Could not open session workspace: %s", ws_exc)

        # Ensure KB is available for this tenant before routing.
        ensure_kb_indexed(self.ctx.settings, self.ctx.stores, tenant_id=session.tenant_id)

        # 1) If uploads are present, ingest + kick off rag tool.
        if upload_paths:
            self.ingest_and_summarize_uploads(session, upload_paths)

        # 2) Route — use hybrid LLM router when enabled, deterministic otherwise.
        if self.ctx.settings.llm_router_enabled:
            decision = route_message_hybrid(
                user_text,
                has_attachments=bool(upload_paths),
                judge_llm=self.ctx.providers.judge,
                history_summary=_summarise_history(session.messages, n=2),
                explicit_force_agent=force_agent,
                llm_confidence_threshold=self.ctx.settings.llm_router_confidence_threshold,
            )
        else:
            decision = route_message(
                user_text, has_attachments=bool(upload_paths), explicit_force_agent=force_agent
            )

        meta = {
            "route": decision.route,
            "router_confidence": decision.confidence,
            "router_reasons": decision.reasons,
            "has_attachments": bool(upload_paths),
            "uploaded_doc_ids": list(session.uploaded_doc_ids),
            "tenant_id": session.tenant_id,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "request_id": session.request_id,
        }

        callbacks = get_langchain_callbacks(
            self.ctx.settings,
            session_id=session.session_id,
            trace_name="chat_turn",
            metadata=meta,
        )
        if extra_callbacks:
            callbacks = list(callbacks) + list(extra_callbacks)

        if decision.route == "BASIC":
            text = run_basic_chat(
                self.ctx.providers.chat,
                messages=session.messages,
                user_text=user_text,
                system_prompt=self._basic_chat_system_prompt,
                callbacks=callbacks,
            )
            session.messages.append(AIMessage(content=text))
            if self.ctx.settings.clear_scratchpad_per_turn:
                session.clear_scratchpad()
            return text

        # 3) Agent — try multi-agent graph, fall back to single GeneralAgent.
        suggested_agent = getattr(decision, "suggested_agent", "")
        try:
            text = self._run_multi_agent_graph(session, user_text, callbacks, suggested_agent=suggested_agent)
        except Exception:
            logger.exception(
                "Unexpected multi-agent graph failure. "
                "Not auto-falling back to avoid masking defects."
            )
            text = (
                "I hit an unexpected internal error in the multi-agent graph and stopped this turn. "
                "Please retry. If the issue persists, check backend logs."
            )
            session.messages.append(AIMessage(content=text))
            if self.ctx.settings.clear_scratchpad_per_turn:
                session.clear_scratchpad()
            return text

        if text is None:
            text = self._run_general_agent_fallback(session, user_text, callbacks)

        # 4) Clear scratchpad after each turn if configured.
        if self.ctx.settings.clear_scratchpad_per_turn:
            session.clear_scratchpad()

        return text

    def _run_multi_agent_graph(
        self,
        session: ChatSession,
        user_text: str,
        callbacks: List[Any],
        *,
        suggested_agent: str = "",
    ) -> Optional[str]:
        """Try to run the multi-agent supervisor graph.

        Returns the final answer text, or None if the graph could not
        be built (e.g. LLM doesn't support tool calling).
        """
        try:
            from agentic_chatbot.graph import build_multi_agent_graph  # noqa: PLC0415
            from agentic_chatbot.graph.builder import build_initial_state  # noqa: PLC0415
            from agentic_chatbot.agents.agent_registry import AgentRegistry  # noqa: PLC0415

            registry = AgentRegistry(self.ctx.settings)

            graph = build_multi_agent_graph(
                chat_llm=self.ctx.providers.chat,
                judge_llm=self.ctx.providers.judge,
                settings=self.ctx.settings,
                stores=self.ctx.stores,
                session=session,
                callbacks=callbacks,
                registry=registry,
            )

            initial_state = build_initial_state(session, user_text, suggested_agent=suggested_agent)

            result = graph.invoke(
                initial_state,
                config={
                    "callbacks": callbacks,
                    "recursion_limit": 50,
                },
            )

            # Sync graph state back to session
            session.messages = list(result.get("messages", session.messages))
            scratchpad = result.get("scratchpad")
            if isinstance(scratchpad, dict):
                session.scratchpad.update(scratchpad)

            # Extract the final answer
            text = result.get("final_answer", "")
            if not text:
                # Fall back to the last AI message content
                for m in reversed(result.get("messages", [])):
                    if isinstance(m, AIMessage) and getattr(m, "content", None):
                        text = m.content
                        break

            return text or "I wasn't able to produce an answer."

        except Exception as e:
            if _is_graph_capability_error(e):
                logger.warning(
                    "Multi-agent graph unavailable due to capability/config issue; "
                    "falling back to GeneralAgent: %s", e,
                    exc_info=True,
                )
                return None
            raise

    def _run_general_agent_fallback(
        self,
        session: ChatSession,
        user_text: str,
        callbacks: List[Any],
    ) -> str:
        """Run the original single-agent path (GeneralAgent with rag_agent_tool)."""
        tools = self._build_tools(session)
        text, msgs, stats = run_general_agent(
            self.ctx.providers.chat,
            tools=tools,
            messages=session.messages,
            user_text=user_text,
            system_prompt=self._general_agent_system_prompt,
            callbacks=callbacks,
            max_steps=self.ctx.settings.max_agent_steps,
            max_tool_calls=self.ctx.settings.max_tool_calls,
        )
        session.messages = msgs
        return text
