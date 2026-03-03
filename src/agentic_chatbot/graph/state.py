"""Shared state schema for the multi-agent LangGraph.

AgentState flows through every node in the graph. The ``messages`` key
uses the built-in ``add_messages`` reducer so messages are appended, never
overwritten. ``rag_results`` uses a custom reducer that supports both:
1) parallel append from workers, and
2) explicit clear after synthesis.
"""
from __future__ import annotations

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


def merge_rag_results(
    current: List[Dict[str, Any]] | None,
    new: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Reducer for rag_results with support for explicit clear markers.

    Workers append normal result dicts. The synthesizer can clear prior results
    by emitting a marker item: ``{"__clear__": True}``.
    """
    cur = list(current or [])
    nxt = list(new or [])
    has_clear = any(isinstance(item, dict) and item.get("__clear__") for item in nxt)
    if has_clear:
        return [item for item in nxt if not (isinstance(item, dict) and item.get("__clear__"))]
    return cur + nxt


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

    # Parallel RAG workers append here; synthesizer clears via {"__clear__": True}.
    rag_results: Annotated[List[Dict[str, Any]], merge_rag_results] = []

    # Final output text
    final_answer: str = ""
