from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .graph_state import AgentState


SUPERVISOR_PROMPT = """You are a supervisor coordinating specialist agents.

Return JSON only:
{
  "next_agent": "rag_agent|utility_agent|parallel_rag|__end__",
  "reason": "short reason",
  "direct_answer": "optional if __end__"
}

Routing guidance:
- utility_agent: arithmetic or list-doc requests.
- parallel_rag: explicit compare/diff between multiple docs.
- rag_agent: document evidence, compliance, citations.
- __end__: if you already have enough answer text in context.
"""
_SUPERVISOR_PROMPT = SUPERVISOR_PROMPT


def _latest_human_text(messages: List[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "content", None):
            return str(m.content)
    return ""


def _fallback_route(user_text: str) -> Dict[str, Any]:
    lower = user_text.lower()
    if any(k in lower for k in ["calculate", "%", "math", "list indexed", "list docs"]):
        return {"next_agent": "utility_agent", "reason": "heuristic_utility"}
    if any(k in lower for k in ["compare", "difference", "diff", "both documents", "v1", "v2"]):
        return {"next_agent": "parallel_rag", "reason": "heuristic_parallel"}
    return {"next_agent": "rag_agent", "reason": "heuristic_rag"}


def make_supervisor_node(
    chat_llm: Any,
    callbacks: list | None = None,
    max_loops: int = 5,
    system_prompt: str = SUPERVISOR_PROMPT,
) -> Callable[[AgentState], Dict[str, Any]]:
    callbacks = callbacks or []
    loop_counter = 0

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        nonlocal loop_counter
        loop_counter += 1

        if loop_counter > max_loops:
            return {
                "next_agent": "__end__",
                "final_answer": "Stopping after max supervisor loops.",
            }

        messages = list(state.get("messages", []))
        user_text = _latest_human_text(messages)

        resp = chat_llm.invoke(
            [SystemMessage(content=system_prompt)] + messages,
            config={"callbacks": callbacks, "tags": ["demo_supervisor"], "metadata": {"loop": loop_counter}},
        )
        content = getattr(resp, "content", "") or ""

        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None

        if not isinstance(parsed, dict) or parsed.get("next_agent") not in {"rag_agent", "utility_agent", "parallel_rag", "__end__"}:
            parsed = _fallback_route(user_text)

        updates: Dict[str, Any] = {"next_agent": parsed["next_agent"]}

        if parsed["next_agent"] == "__end__":
            answer = parsed.get("direct_answer") or content.strip() or "Completed."
            updates["final_answer"] = answer
            updates["messages"] = [AIMessage(content=answer)]

        return updates

    return supervisor_node
