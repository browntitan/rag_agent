from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.prompting import load_grounded_answer_prompt, render_template
from agentic_chatbot_next.utils.json_utils import coerce_float, extract_json


def _answer_claims_missing_evidence(answer: str, warnings: Sequence[str]) -> bool:
    haystack = " ".join([answer, *warnings]).lower()
    phrases = (
        "no evidence",
        "no evidence available",
        "insufficient evidence",
        "no supporting evidence",
        "cannot provide",
    )
    return any(phrase in haystack for phrase in phrases)


def _best_summary_snippet(text: str) -> str:
    candidates = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" -*#\t")
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("graph ") or lower in {"```", "mermaid"}:
            continue
        if len(line) < 32:
            continue
        candidates.append(line)
    if candidates:
        return candidates[0][:220]

    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 32:
            return sentence[:220]
    return normalized[:220]


def _title_overlap_score(question: str, doc: Document) -> int:
    title = str((doc.metadata or {}).get("title") or "").lower()
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.replace("_", " ")))
    overlap = len(q_terms & t_terms)
    if "architecture" in q_terms and "architecture" in t_terms:
        overlap += 2
    return overlap


def _fallback_grounded_answer(question: str, evidence_docs: Sequence[Document], *, warning: str) -> Dict[str, Any]:
    docs = list(evidence_docs)
    prioritized = sorted(docs, key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    if prioritized and _title_overlap_score(question, prioritized[0]) > 0:
        docs = [doc for doc in prioritized if _title_overlap_score(question, doc) > 0][:4]
    else:
        docs = prioritized[:4]
    if not docs:
        return {
            "answer": (
                "I couldn't confidently answer from the retrieved evidence. "
                "Can you clarify what section or keyword you want me to focus on?"
            ),
            "used_citation_ids": [],
            "followups": ["Can you specify what part of the documents you mean?"],
            "warnings": [warning],
            "confidence_hint": 0.2,
        }

    answer_lines = ["Key details from the retrieved documents:"]
    used_citation_ids: list[str] = []
    for doc in docs:
        metadata = doc.metadata or {}
        citation_id = str(metadata.get("chunk_id") or "")
        snippet = _best_summary_snippet(doc.page_content)
        if not snippet:
            continue
        suffix = f" ({citation_id})" if citation_id else ""
        answer_lines.append(f"- {snippet}{suffix}")
        if citation_id:
            used_citation_ids.append(citation_id)

    if len(answer_lines) == 1:
        answer_lines.append("- The retrieved evidence did not contain enough descriptive text to summarize cleanly.")

    return {
        "answer": "\n".join(answer_lines),
        "used_citation_ids": used_citation_ids,
        "followups": [],
        "warnings": [warning],
        "confidence_hint": 0.45 if used_citation_ids else 0.25,
    }


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
            payload = {
                "answer": obj.get("answer", "").strip(),
                "used_citation_ids": [str(item) for item in (obj.get("used_citation_ids") or []) if str(item)],
                "followups": [str(item) for item in (obj.get("followups") or []) if str(item)],
                "warnings": [str(item) for item in (obj.get("warnings") or []) if str(item)],
                "confidence_hint": coerce_float(obj.get("confidence_hint"), default=0.5),
            }
            if docs and _answer_claims_missing_evidence(payload["answer"], payload["warnings"]):
                return _fallback_grounded_answer(question, docs, warning="LLM_NO_EVIDENCE_OVERRIDE")
            return payload
    except Exception:
        pass

    return _fallback_grounded_answer(question, docs, warning="LLM_JSON_PARSE_FAILED")
