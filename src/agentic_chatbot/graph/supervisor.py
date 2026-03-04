"""Supervisor node — LLM-based router that coordinates specialist agents.

The supervisor receives the full conversation history and decides which
specialist agent to hand off to next (or answers directly).  After each
agent finishes, control returns to the supervisor for the next decision.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, SystemMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)

_VALID_AGENTS = {"rag_agent", "utility_agent", "parallel_rag", "__end__"}


def _build_supervisor_prompt(settings: Settings) -> str:
    """Load the supervisor system prompt from skills or use the default."""
    # Lazy import to avoid circular dependency at module level
    from agentic_chatbot.rag.skills import load_supervisor_skills  # noqa: PLC0415

    return load_supervisor_skills(settings)


def _parse_supervisor_response(content: str) -> Dict[str, Any]:
    """Extract the routing decision from the supervisor LLM response.

    Expects a JSON block with at least ``next_agent``.  Falls back to
    heuristic keyword matching if JSON parsing fails.
    """
    # Try to extract JSON from the response
    text = content.strip()

    # Try full response as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "next_agent" in data:
            agent = data["next_agent"]
            if agent not in _VALID_AGENTS:
                logger.warning("Supervisor returned unknown agent %r, defaulting to rag_agent", agent)
                data["next_agent"] = "rag_agent"
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from a markdown code block
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                data = json.loads(text[start:end].strip())
                if isinstance(data, dict) and "next_agent" in data:
                    agent = data["next_agent"]
                    if agent not in _VALID_AGENTS:
                        data["next_agent"] = "rag_agent"
                    return data
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Heuristic fallback: look for keywords
    lower = text.lower()
    if any(kw in lower for kw in ("parallel", "compare", "both documents", "compare these")):
        return {"next_agent": "parallel_rag", "reasoning": "keyword_fallback"}
    if any(kw in lower for kw in ("document", "clause", "requirement", "search", "citation", "policy", "contract")):
        return {"next_agent": "rag_agent", "reasoning": "keyword_fallback"}
    if any(kw in lower for kw in ("calcul", "math", "memory", "remember", "list document", "list indexed")):
        return {"next_agent": "utility_agent", "reasoning": "keyword_fallback"}

    # Default: answer directly
    return {"next_agent": "__end__", "direct_answer": text, "reasoning": "no_json_found"}


def make_supervisor_node(
    chat_llm: Any,
    settings: Settings,
    callbacks: Optional[List[Any]] = None,
    max_loops: int = 5,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create the supervisor node function.

    The returned callable takes ``AgentState`` and returns a partial
    state update dict.
    """
    system_prompt = _build_supervisor_prompt(settings)

    loop_count = 0

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        nonlocal loop_count

        loop_count += 1

        # Demo-mode stabilization: once a specialist has produced a non-empty
        # assistant message for the current user turn, terminate immediately
        # instead of re-routing through extra supervisor loops.
        if state.get("demo_mode", False):
            messages = state.get("messages", [])
            last_human_idx = -1
            for idx, msg in enumerate(messages):
                if getattr(msg, "type", "") == "human":
                    last_human_idx = idx
            if last_human_idx >= 0:
                for ai_idx in range(len(messages) - 1, -1, -1):
                    msg = messages[ai_idx]
                    if getattr(msg, "type", "") != "ai":
                        continue
                    content = str(getattr(msg, "content", "") or "").strip()
                    if not content:
                        continue
                    if ai_idx > last_human_idx:
                        return {"next_agent": "__end__", "final_answer": content}
                    break

        # Safety: force end after max loops
        if loop_count > max_loops:
            logger.warning("Supervisor hit max loops (%d), forcing __end__", max_loops)
            last_content = ""
            for m in reversed(state.get("messages", [])):
                if isinstance(m, AIMessage) and getattr(m, "content", None):
                    last_content = m.content
                    break
            return {
                "next_agent": "__end__",
                "final_answer": last_content or "I've completed my analysis.",
            }

        # Build the supervisor messages
        supervisor_msgs: list = [SystemMessage(content=system_prompt)]

        # Include conversation history
        for m in state.get("messages", []):
            supervisor_msgs.append(m)

        # If parallel RAG results exist, inject them as context
        rag_results = state.get("rag_results", [])
        if rag_results:
            results_text = _format_rag_results(rag_results)
            supervisor_msgs.append(
                SystemMessage(content=f"## Previous RAG Results\n\n{results_text}")
            )

        # Invoke the supervisor LLM
        resp = chat_llm.invoke(
            supervisor_msgs,
            config={
                "callbacks": callbacks or [],
                "tags": ["supervisor"],
                "metadata": {"node": "supervisor", "loop": loop_count},
            },
        )

        content = getattr(resp, "content", str(resp))
        parsed = _parse_supervisor_response(content)

        updates: Dict[str, Any] = {
            "next_agent": parsed["next_agent"],
        }

        # If routing to __end__, set the final answer
        if parsed["next_agent"] == "__end__":
            direct = parsed.get("direct_answer", "")
            if direct:
                updates["final_answer"] = direct
                updates["messages"] = [AIMessage(content=direct)]
            else:
                # Use the raw LLM response as the answer
                updates["final_answer"] = content
                updates["messages"] = [AIMessage(content=content)]

        # If routing to parallel_rag, set the sub-tasks
        if parsed["next_agent"] == "parallel_rag":
            sub_tasks = parsed.get("rag_sub_tasks", [])
            if sub_tasks:
                # Ensure worker IDs
                for i, task in enumerate(sub_tasks):
                    if not task.get("worker_id"):
                        task["worker_id"] = f"rag_worker_{i}"
                updates["rag_sub_tasks"] = sub_tasks
            else:
                # No sub-tasks specified — fall back to single RAG
                logger.warning("Supervisor chose parallel_rag but no sub-tasks, falling back to rag_agent")
                updates["next_agent"] = "rag_agent"

        return updates

    return supervisor_node


def _format_rag_results(results: List[Dict[str, Any]]) -> str:
    """Format collected RAG results into readable text for the supervisor."""
    parts = []
    for r in results:
        worker_id = r.get("worker_id", "unknown")
        contract = r.get("contract", {})
        answer = contract.get("answer", "(no answer)")
        confidence = contract.get("confidence", 0.0)
        citations_count = len(contract.get("citations", []))
        warnings = contract.get("warnings", [])

        part = (
            f"### Worker: {worker_id}\n"
            f"**Answer:** {answer[:500]}{'...' if len(answer) > 500 else ''}\n"
            f"**Confidence:** {confidence:.2f} | **Citations:** {citations_count}\n"
        )
        if warnings:
            part += f"**Warnings:** {', '.join(str(w) for w in warnings)}\n"
        parts.append(part)

    return "\n---\n".join(parts)
