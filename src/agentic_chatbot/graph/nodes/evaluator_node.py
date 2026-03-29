"""Evaluator node — grades agent output against concrete criteria before returning.

Implements the Generator-Evaluator pattern from Anthropic's harness design:
a separate LLM evaluates the agent's answer to catch hallucination,
off-topic responses, and missing citations *before* the user sees it.

The evaluator sits between agent output and END in the graph. On failure,
it can bounce the conversation back to the supervisor for one retry with
a different agent or strategy. Max 1 retry to prevent infinite loops.

Reference: https://www.anthropic.com/engineering/harness-design-long-running-apps
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import AIMessage

from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)

# Criteria the evaluator grades against.  Each criterion maps to a
# concrete, gradable description — not a vague quality judgment.
EVALUATION_CRITERIA = {
    "relevance": "Does the answer directly address the user's question? A relevant answer focuses on what was asked, not tangential information.",
    "evidence": "Are factual claims backed by cited chunks (e.g. doc_id#chunk_id)? An answer without citations for factual claims fails this criterion.",
    "completeness": "Are there obvious gaps or unanswered parts of the question? A complete answer addresses all parts of a multi-part question.",
    "accuracy": "Do the cited chunks actually support the claims made? The answer should not contradict its own sources.",
}

_EVAL_PROMPT_TEMPLATE = """\
You are a strict QA evaluator. Grade this agent answer against each criterion.

## User Question
{question}

## Agent Answer
{answer}

## Criteria
{criteria}

## Instructions
For each criterion, output PASS or FAIL with a brief reason (one sentence).
Then output an overall verdict.

Return ONLY valid JSON in this exact format:
{{"pass": true/false, "failures": ["criterion: reason", ...], "suggestion": "what to improve if failed"}}

If all criteria pass, set "pass": true and "failures": [].
Be strict but fair — a partial answer that acknowledges gaps can still pass."""


def make_evaluator_node(
    llm: Any,
    *,
    criteria: Optional[Dict[str, str]] = None,
) -> Callable:
    """Create the evaluator node function.

    Args:
        llm: The LLM to use for evaluation. Can be a cheaper/faster model
             (e.g. Haiku) since evaluation is a simpler task than generation.
        criteria: Optional custom criteria dict. Defaults to EVALUATION_CRITERIA.

    Returns:
        A callable ``evaluator_node(state) -> dict`` suitable for use as
        a LangGraph node.
    """
    eval_criteria = criteria or EVALUATION_CRITERIA

    def evaluator_node(state: AgentState) -> Dict[str, Any]:
        # Only evaluate RAG/parallel_rag outputs — utility and data_analyst
        # produce deterministic/tool-driven results that don't need LLM QA.
        last_agent = state.get("next_agent", "")
        if last_agent not in ("rag_agent", "parallel_rag", "rag_synthesizer", "supervisor"):
            return {}

        final_answer = state.get("final_answer", "")
        if not final_answer or len(final_answer.strip()) < 20:
            # Too short to evaluate, or empty — skip
            return {}

        # Don't evaluate more than once per turn
        eval_count = state.get("eval_retry_count", 0)
        if eval_count >= 1:
            logger.info("Evaluator: already retried once, accepting answer as-is")
            return {"evaluation_result": "pass"}

        # Extract the user's question from the last HumanMessage
        question = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break

        if not question:
            return {"evaluation_result": "pass"}

        # Build the evaluation prompt
        criteria_text = "\n".join(
            f"- **{name}**: {desc}" for name, desc in eval_criteria.items()
        )
        eval_prompt = _EVAL_PROMPT_TEMPLATE.format(
            question=question[:2000],
            answer=final_answer[:3000],
            criteria=criteria_text,
        )

        try:
            result = llm.invoke(eval_prompt)
            content = result.content if hasattr(result, "content") else str(result)

            # Parse JSON from the response
            parsed = _parse_eval_response(content)

            if parsed["pass"]:
                logger.info("Evaluator: PASS")
                return {"evaluation_result": "pass"}
            else:
                failures_str = "; ".join(parsed["failures"][:3])
                suggestion = parsed.get("suggestion", "Try a different search strategy")
                logger.info("Evaluator: FAIL — %s", failures_str)
                return {
                    "evaluation_result": "fail",
                    "eval_retry_count": eval_count + 1,
                    "final_answer": "",  # clear so supervisor re-routes
                    "next_agent": "",  # let supervisor decide fresh
                    "messages": [
                        AIMessage(
                            content=f"[Evaluator] Previous answer was insufficient: {suggestion}. "
                            f"Failures: {failures_str}. Please try a different approach."
                        )
                    ],
                }
        except Exception as e:
            # If evaluation itself fails, let the answer through rather
            # than blocking the user
            logger.warning("Evaluator failed: %s — accepting answer", e)
            return {"evaluation_result": "pass"}

    return evaluator_node


def _parse_eval_response(content: str) -> Dict[str, Any]:
    """Extract JSON from evaluator LLM response, with fallback."""
    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    import re
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    brace_match = re.search(r"\{[^{}]*\"pass\"[^{}]*\}", content, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: if "FAIL" appears prominently, treat as failure
    if "FAIL" in content.upper() and "PASS" not in content.upper()[:50]:
        return {"pass": False, "failures": ["Could not parse evaluation"], "suggestion": "Retry"}

    # Default: pass (don't block user on parse failure)
    return {"pass": True, "failures": [], "suggestion": ""}
