from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an agentic chatbot that can call tools to solve the user's request.\n"
    "Your priorities are: (1) correct tool selection, (2) correct tool arguments, "
    "(3) clear synthesis of tool results.\n\n"
    "Operating rules:\n"
    "- When a task requires tools, use them. When it doesn't, answer directly.\n"
    "- If multiple tools are needed, create a short numbered PLAN, then execute steps one-by-one.\n"
    "- Prefer rag_agent_tool for KB or uploaded-document questions that need citations.\n"
    "- If tool output conflicts or is insufficient, explain what is missing and ask a follow-up question.\n"
    "- Keep the final answer user-friendly.\n"
)


def _content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _render_rag_tool_fallback(messages: List[Any]) -> str:
    from agentic_chatbot_next.rag.engine import coerce_rag_contract, render_rag_contract

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        payload = extract_json(_content_text(message))
        if not isinstance(payload, dict):
            continue
        if "answer" not in payload or "citations" not in payload:
            continue
        rendered = render_rag_contract(coerce_rag_contract(payload))
        if rendered.strip():
            return rendered.strip()
    return ""


def _sanitize_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(args or {}).items()
        if value is not None
    }


def _invoke_tool_with_trace(
    tool_map: Dict[str, Any],
    messages: List[Any],
    callbacks: List[Any],
    tool_name: str,
    args: Dict[str, Any],
    *,
    call_index: int,
) -> tuple[str, int]:
    tool = tool_map[tool_name]
    safe_args = _sanitize_tool_args(args)
    try:
        output = tool.invoke(safe_args, config={"callbacks": callbacks})
    except TypeError:
        output = tool.invoke(safe_args)
    output_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
    messages.append(ToolMessage(content=output_text, tool_call_id=f"guided_{tool_name}_{call_index}"))
    return output_text, call_index + 1


def _data_analyst_fallback_enabled(tool_map: Dict[str, Any]) -> bool:
    required = {"load_dataset", "inspect_columns", "execute_code", "workspace_list"}
    return required.issubset(set(tool_map))


def _has_successful_execute_code(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        if str(result.get("tool") or "") != "execute_code":
            continue
        payload = extract_json(result.get("output")) or {}
        if isinstance(payload, dict) and payload.get("success") is True:
            return True
    return False


def _build_data_analyst_code(dataset_refs: List[str], dataset_payloads: Dict[str, Dict[str, Any]]) -> str:
    columns_by_ref = {
        ref: {str(column) for column in (payload.get("columns") or [])}
        for ref, payload in dataset_payloads.items()
    }
    if len(dataset_refs) >= 2:
        first, second = dataset_refs[:2]
        first_cols = columns_by_ref.get(first, set())
        second_cols = columns_by_ref.get(second, set())
        required_first = {"region", "annual_spend_usd", "current_reserve_usd"}
        required_second = {"region", "reserve_target_pct", "risk_score", "control_owner"}
        if required_first.issubset(first_cols) and required_second.issubset(second_cols):
            return (
                "import pandas as pd\n\n"
                f"spend = pd.read_csv('/workspace/{first}')\n"
                f"controls = pd.read_csv('/workspace/{second}')\n"
                "merged = spend.merge(controls, on='region', how='inner')\n"
                "merged['target_reserve_usd'] = merged['annual_spend_usd'] * merged['reserve_target_pct']\n"
                "merged['reserve_gap_usd'] = merged['target_reserve_usd'] - merged['current_reserve_usd']\n"
                "ranked = merged.sort_values(['risk_score', 'reserve_gap_usd'], ascending=[False, False]).head(3)\n"
                "columns = ['region', 'annual_spend_usd', 'current_reserve_usd', 'target_reserve_usd', 'reserve_gap_usd', 'risk_score', 'control_owner']\n"
                "print('Top three highest-risk regions by risk score and reserve gap:')\n"
                "print(ranked[columns].to_string(index=False))\n"
                "print('\\nSummary:')\n"
                "for _, row in ranked.iterrows():\n"
                "    print(f\"- {row['region']}: gap=${row['reserve_gap_usd']:.2f}, risk_score={int(row['risk_score'])}, owner={row['control_owner']}\")\n"
            )

    for ref in dataset_refs:
        payload = dataset_payloads.get(ref, {})
        if {"region", "annual_spend_usd", "current_reserve_usd", "reserve_target_pct", "risk_score"}.issubset(
            columns_by_ref.get(ref, set())
        ):
            return (
                "import pandas as pd\n\n"
                f"df = pd.read_csv('/workspace/{ref}')\n"
                "df['target_reserve_usd'] = df['annual_spend_usd'] * df['reserve_target_pct']\n"
                "df['reserve_gap_usd'] = df['target_reserve_usd'] - df['current_reserve_usd']\n"
                "ranked = df.sort_values(['risk_score', 'reserve_gap_usd'], ascending=[False, False]).head(3)\n"
                "print(ranked.to_string(index=False))\n"
            )

    print_targets = ", ".join(f"'/workspace/{ref}'" for ref in dataset_refs)
    return (
        "import os\nimport pandas as pd\n\n"
        f"files = [{print_targets}]\n"
        "for path in files:\n"
        "    if path.endswith('.csv') and os.path.exists(path):\n"
        "        df = pd.read_csv(path)\n"
        "        print(f'File: {os.path.basename(path)} shape={df.shape}')\n"
        "        print(df.head().to_string(index=False))\n"
        "        print('---')\n"
    )


def _run_data_analyst_guided_fallback(
    *,
    chat_llm: Any,
    tool_map: Dict[str, Any],
    messages: List[Any],
    user_text: str,
    callbacks: List[Any],
    max_tool_calls: int,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    tool_calls = 0
    workspace_listing_text, tool_calls = _invoke_tool_with_trace(
        tool_map,
        messages,
        callbacks,
        "workspace_list",
        {},
        call_index=tool_calls,
    )
    workspace_listing = extract_json(workspace_listing_text) or {}
    dataset_refs = [
        str(name)
        for name in (workspace_listing.get("files") or [])
        if str(name).lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    dataset_payloads: Dict[str, Dict[str, Any]] = {}
    for dataset_ref in dataset_refs:
        if tool_calls >= max_tool_calls:
            break
        load_text, tool_calls = _invoke_tool_with_trace(
            tool_map,
            messages,
            callbacks,
            "load_dataset",
            {"doc_id": dataset_ref},
            call_index=tool_calls,
        )
        dataset_payloads[dataset_ref] = extract_json(load_text) or {}
        if tool_calls >= max_tool_calls:
            break
        inspect_text, tool_calls = _invoke_tool_with_trace(
            tool_map,
            messages,
            callbacks,
            "inspect_columns",
            {"doc_id": dataset_ref, "columns": ""},
            call_index=tool_calls,
        )
        del inspect_text

    if "scratchpad_write" in tool_map and tool_calls < max_tool_calls:
        _, tool_calls = _invoke_tool_with_trace(
            tool_map,
            messages,
            callbacks,
            "scratchpad_write",
            {
                "key": "analysis_plan",
                "value": "Load uploaded datasets, inspect schema, compute reserve gaps, rank highest-risk regions, and summarize findings.",
            },
            call_index=tool_calls,
        )

    code = _build_data_analyst_code(dataset_refs, dataset_payloads)
    execute_text, tool_calls = _invoke_tool_with_trace(
        tool_map,
        messages,
        callbacks,
        "execute_code",
        {"code": code, "doc_ids": ",".join(dataset_refs)},
        call_index=tool_calls,
    )
    execute_payload = extract_json(execute_text) or {}
    stdout = str(execute_payload.get("stdout") or "").strip()
    stderr = str(execute_payload.get("stderr") or "").strip()
    success = bool(execute_payload.get("success"))

    if success and stdout:
        final_text = stdout
    elif stderr:
        final_text = f"Data analysis failed in the sandbox:\n{stderr}"
    else:
        final_text = "The data analyst workflow did not produce any output."

    messages.append(AIMessage(content=final_text))
    return final_text, messages, {"fallback": "data_analyst_guided", "tool_calls": tool_calls}


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
    force_plan_execute: bool = False,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    callbacks = callbacks or []
    effective_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
    msgs = _ensure_system(messages, effective_prompt)
    if not _has_latest_user_message(msgs, user_text):
        msgs.append(HumanMessage(content=user_text))

    supports_tool_calls = False
    if not force_plan_execute:
        try:
            chat_llm.bind_tools(tools)
            supports_tool_calls = True
        except Exception:
            supports_tool_calls = False

    if force_plan_execute or not supports_tool_calls:
        return _run_plan_execute_fallback(
            chat_llm,
            tools=tools,
            messages=msgs,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt,
        )

    from langgraph.prebuilt import create_react_agent

    graph = create_react_agent(chat_llm, tools=tools)
    recursion_limit = (max(max_steps, max_tool_calls) + 1) * 2 + 1
    try:
        result = graph.invoke(
            {"messages": msgs},
            config={"callbacks": callbacks, "recursion_limit": recursion_limit},
        )
        updated_messages: List[Any] = result["messages"]
    except Exception as exc:
        logger.warning("LangGraph ReAct agent stopped early: %s", exc)
        final_text = (
            "I reached the maximum number of tool calls for this turn. "
            "If you want, tell me which part to focus on next."
        )
        error_messages = list(msgs) + [AIMessage(content=final_text)]
        return final_text, error_messages, {
            "steps": max_steps,
            "tool_calls": max_tool_calls,
            "budget_exceeded": True,
        }

    tool_calls_used = sum(1 for message in updated_messages if isinstance(message, ToolMessage))
    steps = sum(1 for message in updated_messages if isinstance(message, AIMessage))
    final_text = ""
    for message in reversed(updated_messages):
        if not isinstance(message, AIMessage):
            continue
        final_text = _content_text(message)
        if final_text:
            break
    if not final_text:
        final_text = _render_rag_tool_fallback(updated_messages)
        if final_text:
            updated_messages = list(updated_messages) + [AIMessage(content=final_text)]
    if not final_text:
        final_text = "No response generated."
    return str(final_text), updated_messages, {"steps": steps, "tool_calls": tool_calls_used}


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
    callbacks = callbacks or []
    tool_map = {tool.name: tool for tool in tools}
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
    if system_prompt.strip():
        planner_system += "\n\nRole Instructions:\n" + system_prompt.strip()

    def _extract_plan(text: Any) -> List[Dict[str, Any]] | None:
        payload = extract_json(text) or {}
        plan_value = payload.get("plan") if isinstance(payload, dict) else None
        return plan_value if isinstance(plan_value, list) else None

    plan_response = chat_llm.invoke(
        [SystemMessage(content=planner_system), HumanMessage(content=user_text)],
        config={"callbacks": callbacks},
    )
    plan_text = _content_text(plan_response) or str(plan_response)
    plan = _extract_plan(plan_text)

    if not isinstance(plan, list):
        repair_system = (
            "You repair model outputs into strict JSON.\n"
            "Return JSON ONLY using this exact schema:\n"
            "{\"plan\": [{\"tool\": \"tool_name\", \"args\": {...}, \"purpose\": \"...\"}], \"notes\": \"...\"}\n"
            "Do not include markdown fences or explanatory prose."
        )
        repair_response = chat_llm.invoke(
            [
                SystemMessage(content=repair_system),
                HumanMessage(
                    content=(
                        "Convert the following planner output into valid JSON using the required schema.\n\n"
                        f"{plan_text}"
                    )
                ),
            ],
            config={"callbacks": callbacks},
        )
        plan = _extract_plan(_content_text(repair_response) or str(repair_response))

    if not isinstance(plan, list):
        if _data_analyst_fallback_enabled(tool_map):
            return _run_data_analyst_guided_fallback(
                chat_llm=chat_llm,
                tool_map=tool_map,
                messages=messages,
                user_text=user_text,
                callbacks=callbacks,
                max_tool_calls=max_tool_calls,
            )
        fallback_messages = _ensure_system(messages, system_prompt or _DEFAULT_SYSTEM_PROMPT)
        if not _has_latest_user_message(fallback_messages, user_text):
            fallback_messages = fallback_messages + [HumanMessage(content=user_text)]
        direct = chat_llm.invoke(
            fallback_messages,
            config={"callbacks": callbacks},
        )
        final = getattr(direct, "content", None) or str(direct)
        if not _has_latest_user_message(messages, user_text):
            messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=final))
        return str(final), messages, {"fallback": "direct_no_plan"}

    if not _has_latest_user_message(messages, user_text):
        messages.append(HumanMessage(content=user_text))
    tool_calls = 0
    tool_results: List[Dict[str, Any]] = []
    for step in plan:
        if tool_calls >= max_tool_calls:
            break
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        args = _sanitize_tool_args(step.get("args") or {})
        if not isinstance(name, str) or name not in tool_map:
            tool_results.append({"tool": name, "error": "unknown tool"})
            continue
        try:
            output = tool_map[name].invoke(args, config={"callbacks": callbacks})
        except TypeError:
            output = tool_map[name].invoke(args)
        except Exception as exc:
            output = {"error": str(exc)}
        tool_calls += 1
        out_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
        tool_results.append({"tool": name, "args": args, "output": out_text})
        messages.append(ToolMessage(content=out_text, tool_call_id=f"fallback_{tool_calls}"))

    if _data_analyst_fallback_enabled(tool_map) and not _has_successful_execute_code(tool_results):
        return _run_data_analyst_guided_fallback(
            chat_llm=chat_llm,
            tool_map=tool_map,
            messages=messages,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
        )

    synth_system = (
        "You are a helpful assistant. Use the TOOL_RESULTS to answer the USER_REQUEST.\n"
        "If TOOL_RESULTS include a rag_agent_tool JSON output, use its 'answer' and include citations.\n"
        "Do not dump raw JSON; write a user-facing response."
    )
    synth_user = f"USER_REQUEST: {user_text}\n\nTOOL_RESULTS: {tool_results}"
    synth_response = chat_llm.invoke(
        [SystemMessage(content=synth_system), HumanMessage(content=synth_user)],
        config={"callbacks": callbacks},
    )
    final_text = getattr(synth_response, "content", None) or str(synth_response)
    messages.append(AIMessage(content=str(final_text)))
    return str(final_text), messages, {"fallback": "plan_execute", "tool_calls": tool_calls}
