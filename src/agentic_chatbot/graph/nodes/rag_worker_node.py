"""Parallel RAG worker node — runs one RAG agent per sub-task.

Each worker is instantiated via the Send API with its own state slice.
Workers append their results to ``rag_results`` which are merged by the
``operator.add`` reducer in AgentState.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agentic_chatbot.config import Settings
from agentic_chatbot.graph.session_proxy import SessionProxy
from agentic_chatbot.graph.state import AgentState
from agentic_chatbot.rag import KnowledgeStores

logger = logging.getLogger(__name__)


def make_rag_worker_node(
    settings: Settings,
    stores: KnowledgeStores,
    chat_llm: Any,
    judge_llm: Any,
    callbacks: Optional[List[Any]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create a RAG worker node for parallel execution."""

    def rag_worker_node(state: AgentState) -> Dict[str, Any]:
        from agentic_chatbot.rag.agent import run_rag_agent  # noqa: PLC0415
        from agentic_chatbot.graph.nodes.rag_node import _format_conversation_context  # noqa: PLC0415

        # Each worker sees exactly one task (Send ensures single-element list)
        tasks = state.get("rag_sub_tasks", [])
        if not tasks:
            logger.warning("RAG worker received no sub-tasks")
            return {"rag_results": []}

        task = tasks[0]
        worker_id = task.get("worker_id", "unknown")
        query = task.get("query", "")
        preferred_doc_ids = task.get("preferred_doc_ids", [])

        logger.info("RAG worker %s starting: query=%r, doc_scope=%s", worker_id, query[:80], preferred_doc_ids)

        # Each worker gets its own isolated scratchpad
        session_proxy = SessionProxy(
            session_id=state.get("session_id", ""),
            tenant_id=state.get("tenant_id", "local-dev"),
            scratchpad={},
            uploaded_doc_ids=list(state.get("uploaded_doc_ids", [])),
        )

        context = _format_conversation_context(state.get("messages", []))

        try:
            contract = run_rag_agent(
                settings,
                stores,
                llm=chat_llm,
                judge_llm=judge_llm,
                query=query,
                conversation_context=context,
                preferred_doc_ids=preferred_doc_ids,
                must_include_uploads=bool(state.get("uploaded_doc_ids")),
                top_k_vector=settings.rag_top_k_vector,
                top_k_keyword=settings.rag_top_k_keyword,
                max_retries=settings.rag_max_retries,
                session=session_proxy,
                callbacks=callbacks or [],
            )
        except Exception as e:
            logger.warning("RAG worker %s failed: %s", worker_id, e)
            contract = {
                "answer": f"Worker {worker_id} error: {str(e)[:200]}",
                "citations": [],
                "used_citation_ids": [],
                "confidence": 0.0,
                "followups": [],
                "warnings": [f"WORKER_ERROR: {str(e)[:120]}"],
            }

        logger.info("RAG worker %s done, confidence=%.2f", worker_id, contract.get("confidence", 0))

        # Return as a single-element list — operator.add concatenates across workers
        return {
            "rag_results": [{
                "worker_id": worker_id,
                "query": query,
                "doc_scope": preferred_doc_ids,
                "contract": contract,
            }],
        }

    return rag_worker_node
