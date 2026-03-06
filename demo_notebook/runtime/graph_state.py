from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langgraph.graph import MessagesState


def merge_worker_results(cur: List[Dict[str, Any]] | None, new: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    cur = list(cur or [])
    new = list(new or [])
    if any(isinstance(x, dict) and x.get("__clear__") for x in new):
        return [x for x in new if not (isinstance(x, dict) and x.get("__clear__"))]
    return cur + new


class AgentState(MessagesState):
    next_agent: str = ""
    final_answer: str = ""
    rag_tasks: List[Dict[str, Any]] = []
    worker_results: Annotated[List[Dict[str, Any]], merge_worker_results] = []
