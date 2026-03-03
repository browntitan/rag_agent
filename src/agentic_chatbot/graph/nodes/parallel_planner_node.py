"""Parallel planner node — validates sub-tasks before Send fan-out.

The supervisor populates ``rag_sub_tasks`` in the state before routing
here.  This node validates the tasks and assigns worker IDs if missing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)


def parallel_planner_node(state: AgentState) -> Dict[str, Any]:
    """Validate and normalise rag_sub_tasks for fan-out."""
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

    # Ensure every task has a worker_id and required fields
    for i, task in enumerate(tasks):
        if not task.get("worker_id"):
            task["worker_id"] = f"rag_worker_{i}"
        if "query" not in task:
            task["query"] = ""
        if "preferred_doc_ids" not in task:
            task["preferred_doc_ids"] = []

    logger.info("Parallel planner: %d sub-tasks ready for fan-out", len(tasks))

    return {"rag_sub_tasks": tasks}
