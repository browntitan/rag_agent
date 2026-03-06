from __future__ import annotations

from typing import Any, Callable, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .config import NotebookSettings


RAG_SYSTEM_PROMPT = """You are a specialist RAG agent for enterprise documents.

Rules:
1) Use tools before answering when evidence is needed.
2) Prefer resolve_document when document identity is ambiguous.
3) Use search_document/search_all_documents to gather evidence chunks.
4) For contract deltas use diff_documents/compare_clauses.
5) Always cite chunk identifiers inline like (DOC#chunk0001).
6) If evidence is insufficient, say exactly what is missing.
"""
_RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT


def run_rag_agent(
    llm: Any,
    *,
    tools: List[Callable],
    query: str,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    callbacks: list | None = None,
    max_agent_steps: int = 8,
    max_tool_calls: int = 10,
) -> str:
    callbacks = callbacks or []

    graph = create_react_agent(llm, tools=tools)
    recursion_limit = (max(max_agent_steps, max_tool_calls) + 1) * 2 + 1

    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        },
        config={"callbacks": callbacks, "recursion_limit": recursion_limit},
    )

    messages = result.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "content", None):
            return str(m.content)

    return "I could not produce a RAG answer."


def run_rag_agent_for_doc(
    llm: Any,
    *,
    tools: List[Callable],
    query: str,
    doc_id: str,
    system_prompt: str = RAG_SYSTEM_PROMPT,
    callbacks: list | None = None,
    max_agent_steps: int = 6,
    max_tool_calls: int = 8,
) -> str:
    scoped_query = (
        f"Focus your analysis on document id '{doc_id}'. "
        f"Use document-scoped tools and cite chunk ids.\n\nTask: {query}"
    )
    return run_rag_agent(
        llm,
        tools=tools,
        query=scoped_query,
        system_prompt=system_prompt,
        callbacks=callbacks,
        max_agent_steps=max_agent_steps,
        max_tool_calls=max_tool_calls,
    )


def build_rag_answer_callable(
    settings: NotebookSettings,
    llm: Any,
    rag_tools: List[Callable],
    system_prompt: str = RAG_SYSTEM_PROMPT,
    callbacks: list | None = None,
):
    def _answer(query: str) -> str:
        return run_rag_agent(
            llm,
            tools=rag_tools,
            query=query,
            system_prompt=system_prompt,
            callbacks=callbacks,
            max_agent_steps=settings.max_agent_steps,
            max_tool_calls=settings.max_tool_calls,
        )

    return _answer
