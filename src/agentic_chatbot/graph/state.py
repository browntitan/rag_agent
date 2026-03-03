"""Shared state schema for the multi-agent LangGraph.

AgentState flows through every node in the graph.  The ``messages`` key
uses the built-in ``add_messages`` reducer so messages are appended, never
overwritten.  The ``rag_results`` key uses ``operator.add`` so parallel
RAG workers can independently append their results.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List

from langgraph.graph import MessagesState


@dataclass
class RAGResult:
    """Output of a single parallel RAG worker."""

    query: str
    doc_scope: List[str]
    contract: Dict[str, Any]
    worker_id: str = ""


class AgentState(MessagesState):
    """Top-level state for the multi-agent supervisor graph.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]``
    from ``MessagesState``.
    """

    # Identity / session
    session_id: str = ""
    uploaded_doc_ids: List[str] = []

    # Within-turn working memory (no reducer — last-write-wins)
    scratchpad: Dict[str, str] = {}

    # Supervisor routing
    next_agent: str = ""  # rag_agent | utility_agent | parallel_rag | __end__

    # Parallel RAG planner populates these before Send fan-out
    rag_sub_tasks: List[Dict[str, Any]] = []

    # Parallel RAG workers append here (operator.add concatenates)
    rag_results: Annotated[List[Dict[str, Any]], operator.add] = []

    # Final output text
    final_answer: str = ""
