"""RAG agent graph node — wraps the existing run_rag_agent() as a node.

This is a thin adapter: all retrieval logic in rag/agent.py is unchanged.
The node reads its inputs from AgentState and writes results back.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.graph.session_proxy import SessionProxy
from agentic_chatbot.graph.state import AgentState
from agentic_chatbot.rag import KnowledgeStores

logger = logging.getLogger(__name__)


def _extract_latest_query(messages: list) -> str:
    """Walk backwards through messages to find the most recent user query."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "content", None):
            return m.content
    return ""


def _format_conversation_context(messages: list, max_messages: int = 6) -> str:
    """Build a short conversation context string from recent messages."""
    recent = messages[-(max_messages * 2):] if len(messages) > max_messages * 2 else messages
    parts = []
    for m in recent:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if content and role in ("human", "ai"):
            parts.append(f"{role}: {content[:200]}")
    return "\n".join(parts[-max_messages:])


def render_rag_contract(contract: Dict[str, Any]) -> str:
    """Render a RAG contract dict into a human-readable string."""
    ans = contract.get("answer", "")
    citations = contract.get("citations", [])
    used = set(contract.get("used_citation_ids", []))
    warnings = contract.get("warnings", [])
    followups = contract.get("followups", [])

    lines = [ans.strip()]

    if citations:
        lines.append("\nCitations:")
        for c in citations:
            cid = c.get("citation_id", "")
            if used and cid not in used:
                continue
            title = c.get("title", "")
            loc = c.get("location", "")
            lines.append(f"- [{cid}] {title} ({loc})")

    if warnings:
        lines.append("\nWarnings: " + ", ".join(str(w) for w in warnings))

    if followups:
        lines.append("\nFollow-ups:")
        for q in followups:
            lines.append(f"- {q}")

    return "\n".join(lines).strip()


def make_rag_agent_node(
    settings: Settings,
    stores: KnowledgeStores,
    chat_llm: Any,
    judge_llm: Any,
    callbacks: Optional[List[Any]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create the RAG agent node function."""

    def rag_agent_node(state: AgentState) -> Dict[str, Any]:
        from agentic_chatbot.rag.agent import run_rag_agent  # noqa: PLC0415

        query = _extract_latest_query(state.get("messages", []))
        if not query:
            return {
                "messages": [AIMessage(content="No query found to search for.")],
            }

        session_proxy = SessionProxy(
            session_id=state.get("session_id", ""),
            tenant_id=state.get("tenant_id", "local-dev"),
            scratchpad=dict(state.get("scratchpad", {})),
            uploaded_doc_ids=list(state.get("uploaded_doc_ids", [])),
        )

        context = _format_conversation_context(state.get("messages", []))
        preferred = list(state.get("uploaded_doc_ids", []))

        try:
            contract = run_rag_agent(
                settings,
                stores,
                llm=chat_llm,
                judge_llm=judge_llm,
                query=query,
                conversation_context=context,
                preferred_doc_ids=preferred,
                must_include_uploads=bool(preferred),
                top_k_vector=settings.rag_top_k_vector,
                top_k_keyword=settings.rag_top_k_keyword,
                max_retries=settings.rag_max_retries,
                session=session_proxy,
                callbacks=callbacks or [],
            )
        except Exception as e:
            logger.warning("RAG agent node failed: %s", e)
            contract = {
                "answer": f"RAG agent encountered an error: {str(e)[:200]}",
                "citations": [],
                "used_citation_ids": [],
                "confidence": 0.0,
                "followups": [],
                "warnings": [f"RAG_NODE_ERROR: {str(e)[:120]}"],
            }

        rendered = render_rag_contract(contract)

        return {
            "messages": [AIMessage(content=rendered)],
            "scratchpad": session_proxy.scratchpad,
        }

    return rag_agent_node
