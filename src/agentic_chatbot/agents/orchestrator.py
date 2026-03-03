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
    ensure_kb_indexed,
    ingest_paths,
    load_general_agent_skills,
    load_stores,
)
from agentic_chatbot.router import route_message
from agentic_chatbot.tools import calculator, make_list_docs_tool, make_memory_tools, make_rag_agent_tool

from agentic_chatbot.agents.basic_chat import run_basic_chat
from agentic_chatbot.agents.general_agent import run_general_agent
from agentic_chatbot.agents.session import ChatSession

logger = logging.getLogger(__name__)


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
        # Ensure KB is indexed once at startup.
        ensure_kb_indexed(self.ctx.settings, self.ctx.stores)

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
        list_docs_tool = make_list_docs_tool(s, stores)
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
            metadata={"num_files": len(upload_paths)},
        )

        doc_ids = ingest_paths(s, self.ctx.stores, upload_paths, source_type="upload")
        session.uploaded_doc_ids.extend([d for d in doc_ids if d not in session.uploaded_doc_ids])

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
    ) -> str:
        """Process one user turn. Optionally ingest uploads first."""

        upload_paths = upload_paths or []

        # 1) If uploads are present, ingest + kick off rag tool.
        if upload_paths:
            self.ingest_and_summarize_uploads(session, upload_paths)

        # 2) Route.
        decision = route_message(user_text, has_attachments=bool(upload_paths), explicit_force_agent=force_agent)

        meta = {
            "route": decision.route,
            "router_confidence": decision.confidence,
            "router_reasons": decision.reasons,
            "has_attachments": bool(upload_paths),
            "uploaded_doc_ids": list(session.uploaded_doc_ids),
        }

        callbacks = get_langchain_callbacks(
            self.ctx.settings,
            session_id=session.session_id,
            trace_name="chat_turn",
            metadata=meta,
        )

        if decision.route == "BASIC":
            text = run_basic_chat(
                self.ctx.providers.chat,
                messages=session.messages,
                user_text=user_text,
                system_prompt=self._general_agent_system_prompt,
                callbacks=callbacks,
            )
            session.messages.append(AIMessage(content=text))
            if self.ctx.settings.clear_scratchpad_per_turn:
                session.clear_scratchpad()
            return text

        # 3) Agent — try multi-agent graph, fall back to single GeneralAgent.
        text = self._run_multi_agent_graph(session, user_text, callbacks)
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
    ) -> Optional[str]:
        """Try to run the multi-agent supervisor graph.

        Returns the final answer text, or None if the graph could not
        be built (e.g. LLM doesn't support tool calling).
        """
        try:
            from agentic_chatbot.graph import build_multi_agent_graph  # noqa: PLC0415
            from agentic_chatbot.graph.builder import build_initial_state  # noqa: PLC0415

            graph = build_multi_agent_graph(
                chat_llm=self.ctx.providers.chat,
                judge_llm=self.ctx.providers.judge,
                settings=self.ctx.settings,
                stores=self.ctx.stores,
                session=session,
                callbacks=callbacks,
            )

            initial_state = build_initial_state(session, user_text)

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
            logger.warning(
                "Multi-agent graph failed, falling back to GeneralAgent: %s", e,
                exc_info=True,
            )
            return None

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
