from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot.utils.json_utils import extract_json, coerce_float


@dataclass
class Citation:
    citation_id: str
    doc_id: str
    title: str
    source_type: str
    location: str
    snippet: str


def _location(md: Dict[str, Any]) -> str:
    if "page" in md:
        return f"page {md.get('page')}"
    if "start_index" in md:
        return f"char {md.get('start_index')}"
    if "chunk_index" in md:
        return f"chunk {md.get('chunk_index')}"
    return ""


def build_citations(docs: Sequence[Document], *, max_snippet_chars: int = 320) -> List[Citation]:
    out: List[Citation] = []
    for d in docs:
        md = d.metadata or {}
        cid = str(md.get("chunk_id") or "")
        doc_id = str(md.get("doc_id") or "")
        title = str(md.get("title") or "")
        source_type = str(md.get("source_type") or "")
        loc = _location(md)
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "..."
        out.append(Citation(citation_id=cid, doc_id=doc_id, title=title, source_type=source_type, location=loc, snippet=snippet))
    return out


def generate_grounded_answer(
    llm: Any,
    *,
    question: str,
    conversation_context: str,
    evidence_docs: Sequence[Document],
    max_evidence: int = 8,
    callbacks=None,
) -> Dict[str, Any]:
    """Generate an answer grounded in evidence_docs.

    Returns a dict with keys:
      - answer (str)
      - used_citation_ids (list[str])
      - followups (list[str])
      - warnings (list[str])
      - confidence_hint (float)

    If the LLM output cannot be parsed, returns a fallback answer.
    """

    docs = list(evidence_docs)[:max_evidence]

    evidence_pack = []
    for d in docs:
        md = d.metadata or {}
        evidence_pack.append(
            {
                "citation_id": md.get("chunk_id"),
                "title": md.get("title"),
                "location": "page " + str(md.get("page")) if "page" in md else f"chunk {md.get('chunk_index')}",
                "text": d.page_content[:900],
            }
        )

    prompt = (
        "You are a grounded QA assistant.\n"
        "Answer the QUESTION using ONLY the EVIDENCE snippets provided.\n"
        "Rules:\n"
        "- If a claim depends on evidence, cite it inline using (citation_id).\n"
        "- If evidence is insufficient, say what is missing and ask a clarifying question.\n"
        "- Do NOT fabricate document details.\n\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\"answer\": \"...\", \"used_citation_ids\": [""], \"followups\": [""], \"warnings\": [""], \"confidence_hint\": 0.0}\n\n"
        f"QUESTION: {question}\n"
        f"CONVERSATION_CONTEXT: {conversation_context}\n\n"
        f"EVIDENCE: {evidence_pack}"
    )

    callbacks = callbacks or []

    try:
        resp = llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(resp, "content", None) or str(resp)
        obj = extract_json(text)
        if obj and isinstance(obj.get("answer"), str):
            return {
                "answer": obj.get("answer", "").strip(),
                "used_citation_ids": [str(x) for x in (obj.get("used_citation_ids") or []) if str(x)],
                "followups": [str(x) for x in (obj.get("followups") or []) if str(x)],
                "warnings": [str(x) for x in (obj.get("warnings") or []) if str(x)],
                "confidence_hint": coerce_float(obj.get("confidence_hint"), default=0.5),
            }
    except Exception:
        pass

    # Fallback: best-effort answer
    fallback = "I couldn't confidently answer from the retrieved evidence. Can you clarify what section or keyword you want me to focus on?"
    return {
        "answer": fallback,
        "used_citation_ids": [],
        "followups": ["Can you specify what part of the documents you mean?"],
        "warnings": ["LLM_JSON_PARSE_FAILED"],
        "confidence_hint": 0.2,
    }
