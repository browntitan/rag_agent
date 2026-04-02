from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_chatbot.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = (
    "You are an agentic chatbot that can call tools to solve the user's request.\n"
    "Your priorities are: (1) correct tool selection, (2) correct tool arguments, "
    "(3) clear synthesis of tool results.\n\n"
    "Operating rules:\n"
    "- When a task requires tools, use them. When it doesn't, answer directly.\n"
    "- If multiple tools are needed, create a short numbered PLAN (high-level, no hidden reasoning), "
    "then execute steps one-by-one.\n"
    "- Prefer the rag_agent_tool for questions that depend on the KB, uploaded documents, "
    "policies, runbooks, or anything that needs citations.\n"
    "- The rag_agent_tool returns JSON with keys: answer, citations, used_citation_ids, confidence. "
    "Use the 'answer' field and include citations in your final answer. Do NOT dump raw JSON.\n"
    "- If tool output conflicts or is insufficient, explain what is missing and ask a follow-up question.\n"
    "- Keep the final answer user-friendly.\n"
)


def _ensure_system(messages: List[Any], system_prompt: str) -> List[Any]:
    if not messages or not isinstance(messages[0], SystemMessage):
        return [SystemMessage(content=system_prompt)] + list(messages)
    return list(messages)


def _has_latest_user_message(messages: List[Any], user_text: str) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return False
    return str(getattr(last, "content", "") or "").strip() == user_text.strip()


def run_general_agent(
    chat_llm: Any,
    *,
    tools: List[Any],
    messages: List[Any],
    user_text: str,
    system_prompt: str = "",
    callbacks=None,
    max_steps: int = 10,
    max_tool_calls: int = 12,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """Run a tool-calling agent loop using the LangGraph ReAct agent.

    Uses ``langgraph.prebuilt.create_react_agent`` to replace the former
    hand-written while-loop.  The function signature and return type are
    unchanged so all callers (orchestrator, tests) require no modification.

    Returns (final_text, updated_messages, run_stats).
    """
    callbacks = callbacks or []

    effective_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
    msgs = _ensure_system(messages, effective_prompt)
    if not _has_latest_user_message(msgs, user_text):
        msgs.append(HumanMessage(content=user_text))

    # ------------------------------------------------------------------
    # Check tool-calling support.  Some model wrappers raise on bind_tools;
    # fall back to the plan-execute strategy when that happens.
    # ------------------------------------------------------------------
    try:
        chat_llm.bind_tools(tools)
        supports_tool_calls = True
    except Exception:
        supports_tool_calls = False

    if not supports_tool_calls:
        return _run_plan_execute_fallback(
            chat_llm,
            tools=tools,
            messages=msgs,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt,
        )

    # ------------------------------------------------------------------
    # LangGraph ReAct agent
    # ------------------------------------------------------------------
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    graph = create_react_agent(chat_llm, tools=tools)

    # recursion_limit: LangGraph counts every node visit.
    # Each ReAct cycle = 2 visits (agent node → tools node).
    # The final response-only call = 1 visit.
    # Allow enough room for max_steps LLM calls and max_tool_calls tool calls.
    recursion_limit = (max(max_steps, max_tool_calls) + 1) * 2 + 1

    try:
        result = graph.invoke(
            {"messages": msgs},
            config={"callbacks": callbacks, "recursion_limit": recursion_limit},
        )
        updated_msgs: List[Any] = result["messages"]
    except Exception as e:
        # GraphRecursionError (budget exceeded) or unexpected error — return
        # a graceful message rather than crashing.
        logger.warning("LangGraph ReAct agent stopped early: %s", e)
        final_text = (
            "I reached the maximum number of tool calls for this turn. "
            "If you want, tell me which part to focus on next."
        )
        error_msgs = list(msgs) + [AIMessage(content=final_text)]
        return final_text, error_msgs, {
            "steps": max_steps,
            "tool_calls": max_tool_calls,
            "budget_exceeded": True,
        }

    # Count steps and tool calls from the returned message history.
    tool_calls_used = sum(1 for m in updated_msgs if isinstance(m, ToolMessage))
    steps = sum(1 for m in updated_msgs if isinstance(m, AIMessage))

    # Final answer is the content of the last AIMessage.
    final_ai = next((m for m in reversed(updated_msgs) if isinstance(m, AIMessage)), None)
    final_text = (
        (getattr(final_ai, "content", None) or str(final_ai))
        if final_ai
        else "No response generated."
    )

    return str(final_text), updated_msgs, {"steps": steps, "tool_calls": tool_calls_used}


# ---------------------------------------------------------------------------
# Plan-execute fallback (used when the LLM does not support native tool calls)
# ---------------------------------------------------------------------------

def _run_plan_execute_fallback(
    chat_llm: Any,
    *,
    tools: List[Any],
    messages: List[Any],
    user_text: str,
    callbacks=None,
    max_tool_calls: int = 12,
    system_prompt: str = "",
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """Fallback when the LLM wrapper does not support tool-calling.

    Strategy:
      1) Ask the model to output a JSON plan of tool calls.
      2) Execute the tools deterministically.
      3) Ask the model to synthesize the final answer from tool outputs.
    """

    callbacks = callbacks or []
    tool_map = {t.name: t for t in tools}

    planner_system = (
        "You are a planning assistant. You cannot call tools directly.\n"
        "Produce a tool plan as JSON ONLY.\n"
        "Allowed tools: " + ", ".join(tool_map.keys()) + "\n\n"
        "Return JSON in this schema:\n"
        "{\"plan\": [{\"tool\": \"tool_name\", \"args\": {...}, \"purpose\": \"...\"}], \"notes\": \"...\"}\n\n"
        "Rules:\n"
        "- Use 0 tools if not needed.\n"
        "- Keep args minimal and valid JSON.\n"
        "- Prefer rag_agent_tool when citations or documents are involved.\n"
    )

    plan_resp = chat_llm.invoke(
        [SystemMessage(content=planner_system), HumanMessage(content=user_text)],
        config={"callbacks": callbacks},
    )
    plan_text = getattr(plan_resp, "content", None) or str(plan_resp)
    plan_obj = extract_json(plan_text) or {}
    plan = plan_obj.get("plan") if isinstance(plan_obj, dict) else None
    if not isinstance(plan, list):
        # Fallback to direct answer
        fallback_msgs = _ensure_system(messages, system_prompt or _DEFAULT_SYSTEM_PROMPT)
        direct = chat_llm.invoke(fallback_msgs + [HumanMessage(content=user_text)], config={"callbacks": callbacks})
        final = getattr(direct, "content", None) or str(direct)
        messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=final))
        return str(final), messages, {"fallback": "direct_no_plan"}

    # Execute plan
    messages.append(HumanMessage(content=user_text))
    tool_calls = 0
    tool_results: List[Dict[str, Any]] = []

    for step in plan:
        if tool_calls >= max_tool_calls:
            break
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        args = step.get("args") or {}
        if not isinstance(name, str) or name not in tool_map:
            tool_results.append({"tool": name, "error": "unknown tool"})
            continue
        try:
            out = tool_map[name].invoke(args, config={"callbacks": callbacks})
        except TypeError:
            out = tool_map[name].invoke(args)
        except Exception as e:
            out = {"error": str(e)}
        tool_calls += 1
        if isinstance(out, (dict, list)):
            out_str = json.dumps(out, ensure_ascii=False)
        else:
            out_str = str(out)
        tool_results.append({"tool": name, "args": args, "output": out_str})
        messages.append(ToolMessage(content=out_str, tool_call_id=f"fallback_{tool_calls}"))

    # Synthesize
    synth_system = (
        "You are a helpful assistant. Use the TOOL_RESULTS to answer the USER_REQUEST.\n"
        "If TOOL_RESULTS include a rag_agent_tool JSON output, use its 'answer' and include citations.\n"
        "Do not dump raw JSON; write a user-facing response."
    )
    synth_user = f"USER_REQUEST: {user_text}\n\nTOOL_RESULTS: {tool_results}"
    synth = chat_llm.invoke(
        [SystemMessage(content=synth_system), HumanMessage(content=synth_user)],
        config={"callbacks": callbacks},
    )
    final_text = getattr(synth, "content", None) or str(synth)
    messages.append(AIMessage(content=str(final_text)))
    return str(final_text), messages, {"fallback": "plan_execute", "tool_calls": tool_calls}
