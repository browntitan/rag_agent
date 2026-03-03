from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document

from agentic_chatbot.utils.json_utils import extract_json, coerce_int


@dataclass
class GradedChunk:
    doc: Document
    relevance: int  # 0-3
    reason: str


def _heuristic_relevance(question: str, text: str) -> int:
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", text.lower()))
    overlap = len(q_terms & t_terms)
    if overlap >= 10:
        return 3
    if overlap >= 5:
        return 2
    if overlap >= 2:
        return 1
    return 0


def grade_chunks(
    judge_llm: Any,
    *,
    question: str,
    chunks: Sequence[Document],
    max_chunks: int = 12,
    callbacks=None,
) -> List[GradedChunk]:
    """Grade retrieved chunks for relevance using an LLM judge.

    Returns a list of GradedChunk (doc + relevance score 0..3).
    """

    selected = list(chunks)[:max_chunks]
    items = []
    for d in selected:
        md = d.metadata or {}
        cid = md.get("chunk_id") or f"{md.get('doc_id')}#chunk{md.get('chunk_index')}"
        title = md.get("title", "")
        location = "page " + str(md.get("page")) if "page" in md else f"chunk {md.get('chunk_index')}"
        snippet = d.page_content
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        items.append({"chunk_id": str(cid), "title": str(title), "location": str(location), "text": snippet})

    prompt = (
        "You are a retrieval relevance grader.\n"
        "Given a QUESTION and a list of CHUNKS, assign each chunk a relevance score:\n"
        "3 = directly answers the question or contains key required facts\n"
        "2 = partially relevant / useful supporting information\n"
        "1 = tangentially related\n"
        "0 = not relevant\n\n"
        "Return ONLY valid JSON in this exact schema:\n"
        "{\"grades\": [{\"chunk_id\": \"...\", \"relevance\": 0, \"reason\": \"...\"}, ...]}\n\n"
        f"QUESTION: {question}\n\n"
        f"CHUNKS: {items}"
    )

    callbacks = callbacks or []

    try:
        resp = judge_llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(resp, "content", None) or str(resp)
        obj = extract_json(text)
        if obj and isinstance(obj.get("grades"), list):
            grade_map: Dict[str, Tuple[int, str]] = {}
            for g in obj["grades"]:
                if not isinstance(g, dict):
                    continue
                cid = str(g.get("chunk_id", ""))
                rel = coerce_int(g.get("relevance"), default=0)
                rel = max(0, min(3, rel))
                reason = str(g.get("reason", ""))[:200]
                if cid:
                    grade_map[cid] = (rel, reason)

            out: List[GradedChunk] = []
            for d in selected:
                md = d.metadata or {}
                cid = str(md.get("chunk_id") or "")
                rel, reason = grade_map.get(cid, (_heuristic_relevance(question, d.page_content), "heuristic"))
                out.append(GradedChunk(doc=d, relevance=rel, reason=reason))
            return out
    except Exception:
        pass

    # Fallback heuristic
    out2: List[GradedChunk] = []
    for d in selected:
        rel = _heuristic_relevance(question, d.page_content)
        out2.append(GradedChunk(doc=d, relevance=rel, reason="heuristic"))
    return out2
