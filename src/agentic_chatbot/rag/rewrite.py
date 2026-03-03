from __future__ import annotations

from typing import Any

from agentic_chatbot.config import Settings
from agentic_chatbot.prompting import DEFAULT_JUDGE_REWRITE_PROMPT, load_judge_rewrite_prompt, render_template
from agentic_chatbot.utils.json_utils import extract_json


def rewrite_query(
    judge_llm: Any,
    *,
    settings: Settings | None = None,
    question: str,
    conversation_context: str,
    attempt: int,
    callbacks=None,
) -> str:
    """Rewrite the query to improve retrieval."""

    template = load_judge_rewrite_prompt(settings) if settings is not None else DEFAULT_JUDGE_REWRITE_PROMPT
    prompt = render_template(
        template,
        {
            "ATTEMPT": attempt,
            "QUESTION": question,
            "CONVERSATION_CONTEXT": conversation_context,
        },
    )

    callbacks = callbacks or []

    try:
        resp = judge_llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(resp, "content", None) or str(resp)
        obj = extract_json(text)
        if obj and isinstance(obj.get("rewritten_query"), str) and obj["rewritten_query"].strip():
            return obj["rewritten_query"].strip()
    except Exception:
        pass

    # Basic fallback: return original question
    return question.strip()
