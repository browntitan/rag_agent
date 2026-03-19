"""Data analyst agent graph node.

Wraps the data analyst ReAct agent (create_react_agent) into a LangGraph node
function following the same pattern as utility_node.py.

Tools available to this agent:
- load_dataset       — load Excel/CSV from KB and inspect schema
- inspect_columns    — detailed per-column statistics
- execute_code       — run Python in Docker sandbox
- calculator         — quick safe arithmetic
- scratchpad_write   — within-turn memory
- scratchpad_read    — within-turn memory
- scratchpad_list    — within-turn memory
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


def make_data_analyst_node(
    chat_llm: Any,
    settings: Settings,
    stores: KnowledgeStores,
    session_proxy: SessionProxy,
    callbacks: Optional[List[Any]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create the data analyst agent node function.

    Args:
        chat_llm:      The chat LLM used for the ReAct agent loop.
        settings:      Application settings (sandbox config, budgets, skills paths).
        stores:        KnowledgeStores for dataset file resolution.
        session_proxy: SessionProxy providing scratchpad access within this turn.
        callbacks:     Optional LangChain/Langfuse callbacks.

    Returns:
        A callable ``data_analyst_node(state: AgentState) -> dict`` suitable
        for use as a LangGraph node.
    """
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415
    from agentic_chatbot.tools.data_analyst_tools import make_data_analyst_tools  # noqa: PLC0415
    from agentic_chatbot.rag.skills import load_data_analyst_skills  # noqa: PLC0415

    data_tools = make_data_analyst_tools(stores, session_proxy, settings=settings)
    system_prompt = load_data_analyst_skills(settings)

    try:
        analyst_graph = create_react_agent(chat_llm, tools=data_tools)
        graph_available = True
    except Exception as exc:
        logger.warning("Could not create data analyst ReAct agent: %s", exc)
        graph_available = False

    def data_analyst_node(state: AgentState) -> Dict[str, Any]:
        """Execute the data analyst ReAct loop for one supervisor-delegated turn."""
        if not graph_available:
            return {
                "messages": [AIMessage(
                    content="Data analyst agent is not available (tool calling not supported by this LLM)."
                )],
            }

        from langchain_core.messages import SystemMessage  # noqa: PLC0415

        msgs: list = []
        if system_prompt:
            msgs.append(SystemMessage(content=system_prompt))
        msgs.extend(state.get("messages", []))

        # Budget: same formula as utility_agent
        recursion_limit = (settings.data_analyst_max_steps + settings.max_tool_calls + 1) * 2 + 1

        try:
            result = analyst_graph.invoke(
                {"messages": msgs},
                config={
                    "callbacks": callbacks or [],
                    "recursion_limit": recursion_limit,
                    "tags": ["data_analyst"],
                    "metadata": {"node": "data_analyst"},
                },
            )

            result_msgs = result.get("messages", [])
            # Extract only the new messages the agent added (after our input)
            new_msgs = result_msgs[len(msgs):]
            if not new_msgs:
                # Fallback: return the last AI message
                for m in reversed(result_msgs):
                    if isinstance(m, AIMessage):
                        new_msgs = [m]
                        break

            return {"messages": new_msgs}

        except Exception as exc:
            logger.warning("Data analyst agent failed: %s", exc)
            return {
                "messages": [AIMessage(content=f"Data analyst encountered an error: {str(exc)[:200]}")],
            }

    return data_analyst_node
