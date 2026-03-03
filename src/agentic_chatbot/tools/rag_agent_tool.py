from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.agent import run_rag_agent
from agentic_chatbot.rag.stores import KnowledgeStores


def _parse_csv(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def make_rag_agent_tool(
    settings: Settings,
    stores: KnowledgeStores,
    *,
    llm: Any,
    judge_llm: Any,
    session: Any,   # ChatSession — passed so the RAG loop agent can use scratchpad
) -> Callable:
    """Return a LangChain tool that wraps the loop-based agentic RAG workflow."""

    @tool
    def rag_agent_tool(
        query: str,
        conversation_context: str = "",
        preferred_doc_ids_csv: str = "",
        must_include_uploads: bool = True,
        top_k_vector: int = 12,
        top_k_keyword: int = 12,
        max_retries: int = 2,
        scratchpad_context_key: str = "",
    ) -> Dict[str, Any]:
        """Answer questions grounded in the KB and uploaded documents.

        This is a loop-based agentic RAG tool — it autonomously selects documents,
        runs multiple search strategies, compares clauses, extracts requirements,
        and synthesises a citation-backed answer.

        Args:
          query:                 The user question or task.
          conversation_context:  Optional recent conversation history for disambiguation.
          preferred_doc_ids_csv: Optional comma-separated doc_ids to constrain search.
          must_include_uploads:  If true, prefer uploaded docs in retrieval.
          top_k_vector:          Vector retrieval depth per search call.
          top_k_keyword:         Keyword/FTS retrieval depth per search call.
          max_retries:           Maximum query rewrite retries (fallback mode only).
          scratchpad_context_key: If set, prepend session.scratchpad[key] to
                                  conversation_context — use to chain context
                                  between consecutive rag_agent_tool calls.

        Returns:
          A dict with keys:
            answer, citations, used_citation_ids, confidence,
            retrieval_summary, followups, warnings
        """
        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)

        # Prepend scratchpad context if requested by the GeneralAgent
        if scratchpad_context_key and scratchpad_context_key in session.scratchpad:
            extra = session.scratchpad[scratchpad_context_key]
            conversation_context = f"{extra}\n\n{conversation_context}".strip()

        # Propagate observability callbacks
        callbacks: List[Any] = []
        try:
            from langchain_core.runnables.config import get_config  # type: ignore
            cfg = get_config() or {}
            callbacks = cfg.get("callbacks") or []
        except Exception:
            callbacks = []

        return run_rag_agent(
            settings,
            stores,
            llm=llm,
            judge_llm=judge_llm,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=must_include_uploads,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            max_retries=max_retries,
            session=session,
            callbacks=callbacks,
        )

    return rag_agent_tool
