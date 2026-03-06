from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .general_agent import run_general_agent
from .graph_state import AgentState
from .rag_agent import RAG_SYSTEM_PROMPT, run_rag_agent, run_rag_agent_for_doc
from .supervisor import SUPERVISOR_PROMPT, make_supervisor_node


UTILITY_PROMPT = """You are a utility specialist.

Use calculator and list_indexed_docs when appropriate and answer clearly.
"""
_UTILITY_PROMPT = UTILITY_PROMPT

SYNTHESIS_SYSTEM_PROMPT = "You are a synthesis assistant."



def _latest_human(messages: List[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "content", None):
            return str(m.content)
    return ""


def _extract_doc_ids_from_query(query: str, available_doc_ids: List[str]) -> List[str]:
    q = query.lower()
    hits = [doc_id for doc_id in available_doc_ids if doc_id.lower() in q]
    if len(hits) >= 2:
        return hits[:4]
    return available_doc_ids[:2]


def build_multi_agent_graph(
    *,
    chat_llm: Any,
    utility_tools: List[Callable],
    rag_tools: List[Callable],
    all_doc_ids: List[str],
    supervisor_prompt: str = SUPERVISOR_PROMPT,
    rag_system_prompt: str = RAG_SYSTEM_PROMPT,
    utility_prompt: str = UTILITY_PROMPT,
    synthesis_system_prompt: str = SYNTHESIS_SYSTEM_PROMPT,
    callbacks: list | None = None,
    max_loops: int = 5,
    max_agent_steps: int = 8,
    max_tool_calls: int = 10,
) -> Any:
    callbacks = callbacks or []

    supervisor_fn = make_supervisor_node(
        chat_llm,
        callbacks=callbacks,
        max_loops=max_loops,
        system_prompt=supervisor_prompt,
    )

    def rag_agent_node(state: AgentState) -> Dict[str, Any]:
        query = _latest_human(list(state.get("messages", [])))
        answer = run_rag_agent(
            chat_llm,
            tools=rag_tools,
            query=query,
            system_prompt=rag_system_prompt,
            callbacks=callbacks,
            max_agent_steps=max_agent_steps,
            max_tool_calls=max_tool_calls,
        )
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}

    def utility_node(state: AgentState) -> Dict[str, Any]:
        query = _latest_human(list(state.get("messages", [])))
        answer, _, _ = run_general_agent(
            chat_llm,
            tools=utility_tools,
            user_text=query,
            system_prompt=utility_prompt,
            history=[],
            callbacks=callbacks,
            max_steps=max_agent_steps,
            max_tool_calls=max_tool_calls,
        )
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}

    def parallel_planner(state: AgentState) -> Dict[str, Any]:
        query = _latest_human(list(state.get("messages", [])))
        doc_ids = _extract_doc_ids_from_query(query, all_doc_ids)
        tasks = [{"doc_id": d, "query": query, "worker_id": f"worker_{i+1}"} for i, d in enumerate(doc_ids)]
        return {"rag_tasks": tasks, "worker_results": [{"__clear__": True}]}

    def rag_worker(state: AgentState) -> Dict[str, Any]:
        task = (state.get("rag_tasks") or [{}])[0]
        doc_id = task.get("doc_id", "")
        query = task.get("query", _latest_human(list(state.get("messages", []))))
        worker_id = task.get("worker_id", "worker")
        answer = run_rag_agent_for_doc(
            chat_llm,
            tools=rag_tools,
            query=query,
            doc_id=doc_id,
            system_prompt=rag_system_prompt,
            callbacks=callbacks,
            max_agent_steps=max(4, max_agent_steps - 2),
            max_tool_calls=max(5, max_tool_calls - 2),
        )
        return {"worker_results": [{"worker_id": worker_id, "doc_id": doc_id, "answer": answer}]}

    def rag_synthesizer(state: AgentState) -> Dict[str, Any]:
        query = _latest_human(list(state.get("messages", [])))
        worker_results = state.get("worker_results", [])
        synthesis_prompt = (
            "Synthesize a single answer from parallel worker outputs. "
            "Keep it concise and preserve citations.\n\n"
            f"Original query: {query}\n\n"
            f"Worker results:\n{worker_results}"
        )
        resp = chat_llm.invoke(
            [SystemMessage(content=synthesis_system_prompt), HumanMessage(content=synthesis_prompt)],
            config={"callbacks": callbacks, "tags": ["demo_parallel_synth"]},
        )
        text = getattr(resp, "content", None) or str(resp)
        return {"messages": [AIMessage(content=text)], "final_answer": text, "worker_results": [{"__clear__": True}]}

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_fn)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("utility_agent", utility_node)
    graph.add_node("parallel_planner", parallel_planner)
    graph.add_node("rag_worker", rag_worker)
    graph.add_node("rag_synthesizer", rag_synthesizer)

    graph.add_edge(START, "supervisor")

    def route_from_supervisor(state: AgentState) -> str:
        nxt = state.get("next_agent", "__end__")
        if nxt == "rag_agent":
            return "rag_agent"
        if nxt == "utility_agent":
            return "utility_agent"
        if nxt == "parallel_rag":
            return "parallel_planner"
        return END

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "utility_agent": "utility_agent",
            "parallel_planner": "parallel_planner",
            END: END,
        },
    )

    graph.add_edge("rag_agent", "supervisor")
    graph.add_edge("utility_agent", "supervisor")

    def fan_out_workers(state: AgentState):
        tasks = state.get("rag_tasks", [])
        if not tasks:
            return [Send("rag_synthesizer", state)]
        sends = []
        for task in tasks:
            s = dict(state)
            s["rag_tasks"] = [task]
            sends.append(Send("rag_worker", s))
        return sends

    graph.add_conditional_edges("parallel_planner", fan_out_workers, ["rag_worker", "rag_synthesizer"])
    graph.add_edge("rag_worker", "rag_synthesizer")
    graph.add_edge("rag_synthesizer", "supervisor")

    return graph.compile()


def build_initial_state(history: List[Any], user_text: str) -> Dict[str, Any]:
    return {
        "messages": list(history) + [HumanMessage(content=user_text)],
        "next_agent": "",
        "final_answer": "",
        "rag_tasks": [],
        "worker_results": [],
    }
