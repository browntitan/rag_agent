"""Utility agent graph node — handles calculator, memory, and list_docs.

This agent is a create_react_agent subgraph with the non-RAG tools.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.graph.session_proxy import SessionProxy
from agentic_chatbot.graph.state import AgentState
from agentic_chatbot.rag import KnowledgeStores

logger = logging.getLogger(__name__)


def _extract_latest_user_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "content", None):
            return str(m.content)
    return ""


def _is_demo_list_docs_query(user_text: str) -> bool:
    lower = user_text.lower()
    return (
        ("list" in lower or "show" in lower)
        and ("document" in lower or "docs" in lower)
    )


def _is_demo_reserve_calc_query(user_text: str) -> bool:
    lower = user_text.lower()
    return (
        "%" in lower
        and "reserve" in lower
        and ("calculate" in lower or "monthly" in lower)
    )


def _format_currency(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"${int(round(value)):,}"
    return f"${value:,.2f}"


def _render_grouped_docs_for_demo(payload_text: str) -> Optional[str]:
    try:
        data = json.loads(payload_text)
    except Exception:
        return None
    if isinstance(data, list):
        groups = {
            "contracts": [],
            "security_compliance": [],
            "runbooks": [],
            "api_references": [],
            "other": [],
        }
        for item in data:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            key = "other"
            lower = title.lower()
            if "runbook" in lower or "playbook" in lower:
                key = "runbooks"
            elif lower.startswith("api_") or "api" in lower:
                key = "api_references"
            elif any(token in lower for token in ("agreement", "contract", "addendum", "schedule", "msa", "dpa")):
                key = "contracts"
            elif any(token in lower for token in ("security", "privacy", "compliance", "control", "incident")):
                key = "security_compliance"
            groups[key].append({"title": title})
    else:
        groups = data.get("groups")
        if not isinstance(groups, dict):
            return None

    sections = [
        ("contracts", "Contracts"),
        ("security_compliance", "Security/Compliance"),
        ("runbooks", "Runbooks"),
        ("api_references", "API References"),
    ]
    lines = ["Indexed documents by category:"]
    for key, label in sections:
        docs = groups.get(key, [])
        if not docs:
            lines.append(f"- {label}: none")
            continue
        titles = ", ".join(item.get("title", "") for item in docs if item.get("title"))
        lines.append(f"- {label}: {titles}")
    return "\n".join(lines)


def _compute_demo_reserve_answer(user_text: str) -> Optional[str]:
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", user_text)
    if not pct_match:
        return None
    percent = float(pct_match.group(1))

    numbers: List[float] = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", user_text):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            continue

    candidates = [num for num in numbers if abs(num - percent) > 1e-6]
    annual = max(candidates) if candidates else None
    if annual is None or annual <= 0:
        return None

    reserve = annual * (percent / 100.0)
    monthly = reserve / 12.0
    return (
        f"Annual risk reserve ({percent:g}% of {_format_currency(annual)}): "
        f"{_format_currency(reserve)}\n"
        f"Monthly reserve: {_format_currency(monthly)}"
    )


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

    list_docs_tool = make_list_docs_tool(settings, stores, session_proxy)
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

        # Demo-mode deterministic shortcuts keep showcase scenarios stable when
        # the model's tool-calling output is incomplete.
        if state.get("demo_mode", False):
            user_text = _extract_latest_user_text(state.get("messages", []))
            if user_text:
                if _is_demo_list_docs_query(user_text):
                    try:
                        grouped = list_docs_tool.invoke({})
                        rendered = _render_grouped_docs_for_demo(grouped)
                        if rendered:
                            return {"messages": [AIMessage(content=rendered)]}
                    except Exception as e:
                        logger.warning("Demo list_docs fallback failed: %s", e)
                if _is_demo_reserve_calc_query(user_text):
                    rendered = _compute_demo_reserve_answer(user_text)
                    if rendered:
                        return {"messages": [AIMessage(content=rendered)]}

        # Build messages for the utility subgraph
        from langchain_core.messages import SystemMessage  # noqa: PLC0415

        msgs = []
        if system_prompt:
            msgs.append(SystemMessage(content=system_prompt))
        msgs.extend(state.get("messages", []))

        # Match GeneralAgent budgeting: account for both LLM-step and tool-call caps.
        recursion_limit = (max(settings.max_agent_steps, settings.max_tool_calls) + 1) * 2 + 1

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
