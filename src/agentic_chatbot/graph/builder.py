"""Multi-agent graph builder — assembles and compiles the StateGraph.

The graph implements a supervisor pattern: a central LLM-based supervisor
routes to specialist agent nodes (RAG, utility, data_analyst) and supports
parallel fan-out via the Send API for multi-document comparison tasks.

Graph topology::

    START → supervisor ──→ rag_agent ──→ supervisor (loop)
                     ├──→ utility_agent ──→ supervisor (loop)
                     ├──→ data_analyst ──→ supervisor (loop)
                     ├──→ parallel_planner ──→ [rag_worker × N] ──→ rag_synthesizer ──→ supervisor
                     └──→ END
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentic_chatbot.agents.session import ChatSession
from agentic_chatbot.config import Settings
from agentic_chatbot.graph.session_proxy import SessionProxy
from agentic_chatbot.graph.state import AgentState
from agentic_chatbot.rag import KnowledgeStores

logger = logging.getLogger(__name__)


def build_multi_agent_graph(
    chat_llm: Any,
    judge_llm: Any,
    settings: Settings,
    stores: KnowledgeStores,
    session: ChatSession,
    callbacks: Optional[List[Any]] = None,
    registry: Any = None,
) -> Any:
    """Build and compile the multi-agent LangGraph.

    Parameters
    ----------
    chat_llm : BaseChatModel
        The chat LLM used by the supervisor and utility agent.
    judge_llm : BaseChatModel
        The judge LLM passed to the RAG agent for grading.
    settings : Settings
        Application settings.
    stores : KnowledgeStores
        Database stores (chunk_store, doc_store, mem_store).
    session : ChatSession
        The current chat session (used to build the SessionProxy).
    callbacks : list, optional
        Langfuse / LangChain callbacks.
    registry : AgentRegistry, optional
        Agent registry for dynamic supervisor prompt generation. When provided,
        the data_analyst node is included only if Docker is available.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke(state, config)``.
    """
    from agentic_chatbot.graph.supervisor import make_supervisor_node
    from agentic_chatbot.graph.nodes.rag_node import make_rag_agent_node
    from agentic_chatbot.graph.nodes.utility_node import make_utility_agent_node
    from agentic_chatbot.graph.nodes.rag_worker_node import make_rag_worker_node
    from agentic_chatbot.graph.nodes.rag_synthesizer_node import make_rag_synthesizer_node
    from agentic_chatbot.graph.nodes.parallel_planner_node import parallel_planner_node
    from agentic_chatbot.graph.nodes.data_analyst_node import make_data_analyst_node

    # Build a session proxy for tool factories.
    # workspace is a reference copy — both ChatSession and SessionProxy point
    # to the same SessionWorkspace object (same host directory), so files
    # written by workspace tools are immediately visible to all consumers.
    session_proxy = SessionProxy(
        session_id=session.session_id,
        tenant_id=session.tenant_id,
        demo_mode=bool(getattr(session, "demo_mode", False)),
        scratchpad=dict(session.scratchpad),
        uploaded_doc_ids=list(session.uploaded_doc_ids),
        workspace=getattr(session, "workspace", None),
    )

    # How many supervisor loops before forcing __end__
    max_loops = getattr(settings, "supervisor_max_loops", 5)

    # Determine if data_analyst is enabled (requires Docker)
    data_analyst_enabled = False
    if registry is not None:
        spec = registry.get("data_analyst")
        data_analyst_enabled = spec is not None and spec.enabled

    # ── Create node functions ──────────────────────────────────────────

    supervisor_fn = make_supervisor_node(
        chat_llm, settings, callbacks=callbacks, max_loops=max_loops, registry=registry,
    )
    rag_agent_fn = make_rag_agent_node(
        settings, stores, chat_llm, judge_llm,
        callbacks=callbacks,
        session_proxy=session_proxy,
    )
    utility_fn = make_utility_agent_node(
        chat_llm, settings, stores, session_proxy, callbacks=callbacks,
    )
    rag_worker_fn = make_rag_worker_node(
        settings, stores, chat_llm, judge_llm, callbacks=callbacks,
    )
    rag_synthesizer_fn = make_rag_synthesizer_node(
        chat_llm, settings=settings, callbacks=callbacks,
    )

    data_analyst_fn = None
    if data_analyst_enabled:
        data_analyst_fn = make_data_analyst_node(
            chat_llm, settings, stores, session_proxy, callbacks=callbacks,
        )

    # ── Build the graph ────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("supervisor", supervisor_fn)
    graph.add_node("rag_agent", rag_agent_fn)
    graph.add_node("utility_agent", utility_fn)
    graph.add_node("parallel_planner", parallel_planner_node)
    graph.add_node("rag_worker", rag_worker_fn)
    graph.add_node("rag_synthesizer", rag_synthesizer_fn)

    if data_analyst_fn is not None:
        graph.add_node("data_analyst", data_analyst_fn)

    # ── Edges ──────────────────────────────────────────────────────────

    # Entry: always start with supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor conditional routing
    def route_from_supervisor(state: AgentState) -> str:
        next_agent = state.get("next_agent", "__end__")
        if next_agent == "rag_agent":
            return "rag_agent"
        elif next_agent == "utility_agent":
            return "utility_agent"
        elif next_agent == "parallel_rag":
            return "parallel_planner"
        elif next_agent == "data_analyst" and data_analyst_fn is not None:
            return "data_analyst"
        elif next_agent == "data_analyst" and data_analyst_fn is None:
            # Docker not available — fall back to rag_agent or end
            logger.warning("Supervisor routed to data_analyst but Docker is unavailable; falling back to rag_agent")
            return "rag_agent"
        else:
            return END

    conditional_edges_map: dict = {
        "rag_agent": "rag_agent",
        "utility_agent": "utility_agent",
        "parallel_planner": "parallel_planner",
        END: END,
    }
    if data_analyst_fn is not None:
        conditional_edges_map["data_analyst"] = "data_analyst"

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        conditional_edges_map,
    )

    # Agent → supervisor (loop back after each agent finishes)
    graph.add_edge("rag_agent", "supervisor")
    graph.add_edge("utility_agent", "supervisor")

    if data_analyst_fn is not None:
        graph.add_edge("data_analyst", "supervisor")

    # Parallel planner → fan-out to rag_workers via Send API
    def fan_out_rag_workers(state: AgentState) -> list:
        tasks = state.get("rag_sub_tasks", [])
        if not tasks:
            # Nothing to fan out — go to synthesizer with empty results
            return [Send("rag_synthesizer", state)]

        max_workers = getattr(settings, "max_parallel_rag_workers", 4)
        tasks_to_run = tasks[:max_workers]

        sends = []
        for task in tasks_to_run:
            worker_state = dict(state)
            worker_state["rag_sub_tasks"] = [task]
            worker_state["rag_results"] = []  # each worker starts clean
            sends.append(Send("rag_worker", worker_state))

        return sends

    graph.add_conditional_edges(
        "parallel_planner",
        fan_out_rag_workers,
        ["rag_worker", "rag_synthesizer"],
    )

    # All rag_workers → rag_synthesizer
    graph.add_edge("rag_worker", "rag_synthesizer")

    # Synthesizer → supervisor (loop back for potential follow-up)
    graph.add_edge("rag_synthesizer", "supervisor")

    # ── Compile ────────────────────────────────────────────────────────

    compiled = graph.compile()
    node_count = 7 if data_analyst_fn is not None else 6
    analyst_note = " + data_analyst" if data_analyst_fn is not None else ""
    logger.info(
        "Multi-agent graph compiled: %d nodes, supervisor + rag_agent + utility + parallel_rag%s",
        node_count,
        analyst_note,
    )

    return compiled


_VALID_SUGGESTED_AGENTS = {"rag_agent", "utility_agent", "parallel_rag", "data_analyst"}


def build_initial_state(
    session: ChatSession,
    user_text: str,
    *,
    suggested_agent: str = "",
) -> dict:
    """Construct the initial AgentState dict from a ChatSession + user input.

    Args:
        session:         Current chat session.
        user_text:       The user's current message.
        suggested_agent: Optional routing hint from the LLM router.
            When set to a valid agent name the supervisor's first loop is
            skipped — the graph enters the named specialist directly.
            Pass ``""`` (default) to let the supervisor decide normally.

    This is a convenience function used by the orchestrator.
    """
    # Pre-seed next_agent only for recognised specialist names.
    next_agent = suggested_agent if suggested_agent in _VALID_SUGGESTED_AGENTS else ""

    return {
        "messages": list(session.messages) + [HumanMessage(content=user_text)],
        "tenant_id": session.tenant_id,
        "session_id": session.session_id,
        "uploaded_doc_ids": list(session.uploaded_doc_ids),
        "demo_mode": bool(getattr(session, "demo_mode", False)),
        "scratchpad": dict(session.scratchpad),
        "next_agent": next_agent,
        "rag_sub_tasks": [],
        "rag_results": [],
        "final_answer": "",
    }
