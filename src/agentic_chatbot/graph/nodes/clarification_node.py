"""Clarification node — asks the user for more information before dispatching.

When the supervisor determines that the user's request is too vague or
ambiguous to route safely (e.g. "summarise the document" with no document
context, or an extremely short query with no uploaded files), it sets:

    needs_clarification=True
    clarification_question="<the question to ask>"

and routes ``next_agent="clarify"``.  This node emits the question as an
``AIMessage``, resets the clarification state flags, and sets
``next_agent="__end__"`` so the graph exits the current turn.

On the **next turn** the user's answer will be part of the conversation
history, giving the supervisor enough context to route properly.

Design rationale
----------------
Using a dedicated node rather than LangGraph ``interrupt()`` keeps the
pattern compatible with the existing stateless API design: each HTTP
request maps to one ``graph.invoke()`` call, and the graph exits cleanly
at the end of every turn. No checkpointer is required.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage

from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION = (
    "Could you provide more details? For example: which document are you "
    "referring to, or what specific aspect would you like me to focus on?"
)


def make_clarification_node() -> Callable[[AgentState], Dict[str, Any]]:
    """Return the clarification node function.

    The returned function is a pure state transformer — it reads
    ``clarification_question`` from the state and emits an ``AIMessage``
    containing that question, then resets the flags and ends the turn.
    """

    def clarification_node(state: AgentState) -> Dict[str, Any]:
        question = (state.get("clarification_question") or "").strip()
        if not question:
            question = _DEFAULT_QUESTION

        logger.info("Clarification node: emitting question: %r", question[:120])

        return {
            "messages": [AIMessage(content=question)],
            "next_agent": "__end__",
            "final_answer": question,
            "needs_clarification": False,
            "clarification_question": "",
        }

    return clarification_node
