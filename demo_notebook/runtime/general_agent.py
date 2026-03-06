from __future__ import annotations

from typing import Any, Callable, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent


GENERAL_SYSTEM_PROMPT = """You are a general orchestration agent.

Rules:
1) Use tools when they help ground the answer.
2) For KB-heavy questions prefer rag_agent_tool.
3) Keep final answers concise and user-facing.
"""
_GENERAL_SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT


def run_general_agent(
    llm: Any,
    *,
    tools: List[Callable],
    user_text: str,
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
    history: List[Any] | None = None,
    callbacks: list | None = None,
    max_steps: int = 8,
    max_tool_calls: int = 10,
) -> Tuple[str, List[Any], dict]:
    callbacks = callbacks or []
    history = list(history or [])

    graph = create_react_agent(llm, tools=tools)
    recursion_limit = (max(max_steps, max_tool_calls) + 1) * 2 + 1

    messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=user_text)]

    result = graph.invoke(
        {"messages": messages},
        config={"callbacks": callbacks, "recursion_limit": recursion_limit},
    )
    out_messages = result.get("messages", messages)

    final_text = ""
    for m in reversed(out_messages):
        if isinstance(m, AIMessage) and getattr(m, "content", None):
            final_text = str(m.content)
            break
    if not final_text:
        final_text = "I could not produce an answer."

    stats = {
        "steps": sum(1 for m in out_messages if isinstance(m, AIMessage)),
        "tool_calls": sum(1 for m in out_messages if isinstance(m, ToolMessage)),
    }
    return final_text, out_messages, stats
