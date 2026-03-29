"""Parallel planner node — validates sub-tasks before Send fan-out.

The supervisor populates ``rag_sub_tasks`` in the state before routing
here.  This node validates the tasks, assigns worker IDs, and enriches
each task with delegation specs (objective, output_format, boundary) so
workers have clear, non-overlapping objectives.

Reference: Anthropic multi-agent research system — "vague instructions
caused duplication — subagents need an objective, output format, guidance
on tools and sources, and clear task boundaries."
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)


def _build_delegation_spec(task: Dict[str, Any], task_index: int, total_tasks: int) -> Dict[str, Any]:
    """Enrich a sub-task with objective, output format, and task boundary.

    These fields guide the RAG worker to produce focused, non-overlapping
    results that the synthesizer can meaningfully combine.
    """
    query = task.get("query", "")
    preferred_docs = task.get("preferred_doc_ids", [])
    doc_scope_str = f"documents {preferred_docs}" if preferred_docs else "all available documents"

    if not task.get("objective"):
        task["objective"] = (
            f"Answer the following question using {doc_scope_str}: {query}. "
            f"Be thorough — this is subtask {task_index + 1} of {total_tasks}."
        )

    if not task.get("output_format"):
        task["output_format"] = (
            "Provide a structured answer with: (1) direct answer to the question, "
            "(2) supporting evidence with chunk citations (doc_id#chunk_id), "
            "(3) confidence level. Do not pad with information not found in the documents."
        )

    if not task.get("boundary"):
        if preferred_docs:
            task["boundary"] = (
                f"Only search within {doc_scope_str}. "
                "Do not cross-reference other documents unless explicitly asked."
            )
        else:
            task["boundary"] = "Search all available documents but stay focused on the query."

    if not task.get("search_strategy"):
        # Infer a sensible default strategy from the query
        q_lower = query.lower()
        if any(kw in q_lower for kw in ("define", "meaning", "what is", "clause", "section")):
            task["search_strategy"] = "keyword"
        elif any(kw in q_lower for kw in ("similar", "related", "like", "context")):
            task["search_strategy"] = "vector"
        else:
            task["search_strategy"] = "hybrid"

    return task


def parallel_planner_node(state: AgentState) -> Dict[str, Any]:
    """Validate, normalise, and enrich rag_sub_tasks for fan-out."""
    tasks: List[Dict[str, Any]] = list(state.get("rag_sub_tasks", []))

    if not tasks:
        # Fallback: if no sub-tasks were specified, create a single task
        # from the latest user query
        query = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage) and getattr(m, "content", None):
                query = m.content
                break
        tasks = [{"query": query, "preferred_doc_ids": [], "worker_id": "rag_worker_0"}]
        logger.info("Parallel planner: no sub-tasks provided, created single task from user query")

    # Check if all tasks have vague short queries — may need clarification
    vague_tasks = [t for t in tasks if len(t.get("query", "").split()) < 3]
    if vague_tasks and len(vague_tasks) == len(tasks):
        logger.info("Parallel planner: all %d tasks have vague queries, requesting clarification", len(tasks))
        return {
            "rag_sub_tasks": tasks,
            "needs_clarification": True,
            "clarification_question": (
                "I'd like to compare documents for you, but I need more specific details. "
                "Could you tell me which aspects to focus on? For example: pricing terms, "
                "obligations, deadlines, or specific clause numbers."
            ),
        }

    # Ensure every task has required fields + enriched delegation specs
    total = len(tasks)
    for i, task in enumerate(tasks):
        if not task.get("worker_id"):
            task["worker_id"] = f"rag_worker_{i}"
        if "query" not in task:
            task["query"] = ""
        if "preferred_doc_ids" not in task:
            task["preferred_doc_ids"] = []
        _build_delegation_spec(task, i, total)

    logger.info("Parallel planner: %d sub-tasks ready for fan-out", len(tasks))

    return {"rag_sub_tasks": tasks}
