"""Utility agent graph node — handles calculator, memory, and list_docs.

This agent is a create_react_agent subgraph with the non-RAG tools.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.graph.session_proxy import SessionProxy
from agentic_chatbot.graph.state import AgentState
from agentic_chatbot.rag import KnowledgeStores

logger = logging.getLogger(__name__)


def make_utility_agent_node(
    chat_llm: Any,
    settings: Settings,
    stores: KnowledgeStores,
    session_proxy: SessionProxy,
    callbacks: Optional[List[Any]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create the utility agent node function.

    The utility agent has: calculator, list_indexed_docs, memory_save,
    memory_load, memory_list.
    """
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415
    from agentic_chatbot.tools import calculator, make_list_docs_tool, make_memory_tools  # noqa: PLC0415
    from agentic_chatbot.rag.skills import load_utility_agent_skills  # noqa: PLC0415

    list_docs_tool = make_list_docs_tool(settings, stores)
    memory_tools = make_memory_tools(stores, session_proxy)
    utility_tools = [calculator, list_docs_tool] + memory_tools

    system_prompt = load_utility_agent_skills(settings)

    # Build the ReAct subgraph
    try:
        utility_graph = create_react_agent(chat_llm, tools=utility_tools)
        graph_available = True
    except Exception as e:
        logger.warning("Could not create utility ReAct agent: %s", e)
        graph_available = False

    def utility_agent_node(state: AgentState) -> Dict[str, Any]:
        if not graph_available:
            return {
                "messages": [AIMessage(content="Utility agent is not available (tool calling not supported by this LLM).")],
            }

        # Build messages for the utility subgraph
        from langchain_core.messages import SystemMessage  # noqa: PLC0415

        msgs = []
        if system_prompt:
            msgs.append(SystemMessage(content=system_prompt))
        msgs.extend(state.get("messages", []))

        recursion_limit = (settings.max_agent_steps + 1) * 2 + 1

        try:
            result = utility_graph.invoke(
                {"messages": msgs},
                config={
                    "callbacks": callbacks or [],
                    "recursion_limit": recursion_limit,
                    "tags": ["utility_agent"],
                },
            )
            # Extract the new messages produced by the utility agent
            result_msgs = result.get("messages", [])
            # Find messages added by the agent (after our input)
            new_msgs = result_msgs[len(msgs):]
            if not new_msgs:
                # Fallback: take the last AIMessage
                for m in reversed(result_msgs):
                    if isinstance(m, AIMessage):
                        new_msgs = [m]
                        break

            return {"messages": new_msgs}

        except Exception as e:
            logger.warning("Utility agent failed: %s", e)
            return {
                "messages": [AIMessage(content=f"Utility agent error: {str(e)[:200]}")],
            }

    return utility_agent_node
