from __future__ import annotations

from typing import Any, Dict, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.prompting import load_grounded_answer_prompt, render_template
from agentic_chatbot_next.utils.json_utils import coerce_float, extract_json


def generate_grounded_answer(
    llm: Any,
    *,
    settings: Any,
    question: str,
    conversation_context: str,
    evidence_docs: Sequence[Document],
    max_evidence: int = 8,
    callbacks=None,
) -> Dict[str, Any]:
    docs = list(evidence_docs)[:max_evidence]
    evidence_pack = []
    for doc in docs:
        metadata = doc.metadata or {}
        evidence_pack.append(
            {
                "citation_id": metadata.get("chunk_id"),
                "title": metadata.get("title"),
                "location": "page " + str(metadata.get("page")) if "page" in metadata else f"chunk {metadata.get('chunk_index')}",
                "text": doc.page_content[:900],
            }
        )
    prompt = render_template(
        load_grounded_answer_prompt(settings),
        {
            "QUESTION": question,
            "CONVERSATION_CONTEXT": conversation_context,
            "EVIDENCE_JSON": evidence_pack,
        },
    )
    callbacks = callbacks or []
    try:
        response = llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(response, "content", None) or str(response)
        obj = extract_json(text)
        if obj and isinstance(obj.get("answer"), str):
            return {
                "answer": obj.get("answer", "").strip(),
                "used_citation_ids": [str(item) for item in (obj.get("used_citation_ids") or []) if str(item)],
                "followups": [str(item) for item in (obj.get("followups") or []) if str(item)],
                "warnings": [str(item) for item in (obj.get("warnings") or []) if str(item)],
                "confidence_hint": coerce_float(obj.get("confidence_hint"), default=0.5),
            }
    except Exception:
        pass

    return {
        "answer": (
            "I couldn't confidently answer from the retrieved evidence. "
            "Can you clarify what section or keyword you want me to focus on?"
        ),
        "used_citation_ids": [],
        "followups": ["Can you specify what part of the documents you mean?"],
        "warnings": ["LLM_JSON_PARSE_FAILED"],
        "confidence_hint": 0.2,
    }
