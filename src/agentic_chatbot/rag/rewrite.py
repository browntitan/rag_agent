from __future__ import annotations

from typing import Any

from agentic_chatbot.utils.json_utils import extract_json


def rewrite_query(
    judge_llm: Any,
    *,
    question: str,
    conversation_context: str,
    attempt: int,
    callbacks=None,
) -> str:
    """Rewrite the query to improve retrieval."""

    prompt = (
        "You are a query rewriting assistant for retrieval.\n"
        "Rewrite the QUESTION into a better search query.\n"
        "- Prefer concrete nouns and key terms\n"
        "- Include synonyms if helpful\n"
        "- Remove filler words\n"
        "Return ONLY valid JSON: {\"rewritten_query\": \"...\"}.\n\n"
        f"ATTEMPT: {attempt}\n"
        f"QUESTION: {question}\n"
        f"CONVERSATION_CONTEXT: {conversation_context}\n"
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
