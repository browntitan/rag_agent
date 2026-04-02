from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from agentic_chatbot_next.prompting import load_judge_grading_prompt, render_template
from agentic_chatbot_next.persistence.postgres.chunks import ScoredChunk
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.utils.json_utils import coerce_int, extract_json

__all__ = [
    "ScoredChunk",
    "GradedChunk",
    "keyword_search",
    "merge_dedupe",
    "retrieve_candidates",
    "vector_search",
    "grade_chunks",
]


@dataclass
class GradedChunk:
    doc: Document
    relevance: int
    reason: str


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}#{metadata.get('chunk_index')}")


def vector_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    tenant_id: str,
    doc_id_filter: Optional[str] = None,
) -> List[ScoredChunk]:
    return stores.chunk_store.vector_search(
        query,
        top_k=top_k,
        doc_id_filter=doc_id_filter,
        tenant_id=tenant_id,
    )


def keyword_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    tenant_id: str,
    doc_id_filter: Optional[str] = None,
) -> List[ScoredChunk]:
    return stores.chunk_store.keyword_search(
        query,
        top_k=top_k,
        doc_id_filter=doc_id_filter,
        tenant_id=tenant_id,
    )


def merge_dedupe(chunks: Sequence[ScoredChunk]) -> List[ScoredChunk]:
    by_key: Dict[str, ScoredChunk] = {}
    for chunk in chunks:
        key = _doc_key(chunk.doc)
        if key not in by_key or chunk.score > by_key[key].score:
            by_key[key] = chunk
    return list(by_key.values())


def retrieve_candidates(
    stores: KnowledgeStores,
    query: str,
    *,
    tenant_id: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    doc_id_filter: Optional[str] = None,
) -> Dict[str, Any]:
    effective_filter = doc_id_filter
    vector_hits = vector_search(
        stores,
        query,
        top_k=top_k_vector,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
    )
    keyword_hits = keyword_search(
        stores,
        query,
        top_k=top_k_keyword,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
    )
    if not effective_filter and preferred_doc_ids:
        vector_hits = [chunk for chunk in vector_hits if (chunk.doc.metadata or {}).get("doc_id") in preferred_doc_ids]
        keyword_hits = [chunk for chunk in keyword_hits if (chunk.doc.metadata or {}).get("doc_id") in preferred_doc_ids]

    if must_include_uploads:
        boosted_vector: List[ScoredChunk] = []
        for chunk in vector_hits:
            if (chunk.doc.metadata or {}).get("source_type") == "upload":
                boosted_vector.append(ScoredChunk(doc=chunk.doc, score=chunk.score + 0.01, method=chunk.method))
            else:
                boosted_vector.append(chunk)
        vector_hits = boosted_vector

    merged = merge_dedupe(vector_hits + keyword_hits)
    return {"vector": vector_hits, "keyword": keyword_hits, "merged": merged}


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
    settings: Any,
    question: str,
    chunks: Sequence[Document],
    max_chunks: int = 12,
    callbacks=None,
) -> List[GradedChunk]:
    selected = list(chunks)[:max_chunks]
    items = []
    for chunk in selected:
        metadata = chunk.metadata or {}
        chunk_id = metadata.get("chunk_id") or f"{metadata.get('doc_id')}#chunk{metadata.get('chunk_index')}"
        title = metadata.get("title", "")
        location = "page " + str(metadata.get("page")) if "page" in metadata else f"chunk {metadata.get('chunk_index')}"
        snippet = chunk.page_content[:800] + ("..." if len(chunk.page_content) > 800 else "")
        items.append({"chunk_id": str(chunk_id), "title": str(title), "location": str(location), "text": snippet})

    prompt = render_template(
        load_judge_grading_prompt(settings),
        {"QUESTION": question, "CHUNKS_JSON": items},
    )
    callbacks = callbacks or []
    try:
        response = judge_llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(response, "content", None) or str(response)
        obj = extract_json(text)
        if obj and isinstance(obj.get("grades"), list):
            grade_map: Dict[str, Tuple[int, str]] = {}
            for grade in obj["grades"]:
                if not isinstance(grade, dict):
                    continue
                chunk_id = str(grade.get("chunk_id", ""))
                relevance = coerce_int(grade.get("relevance"), default=0)
                relevance = max(0, min(3, relevance))
                reason = str(grade.get("reason", ""))[:200]
                if chunk_id:
                    grade_map[chunk_id] = (relevance, reason)
            graded: List[GradedChunk] = []
            for chunk in selected:
                chunk_id = str((chunk.metadata or {}).get("chunk_id") or "")
                relevance, reason = grade_map.get(
                    chunk_id,
                    (_heuristic_relevance(question, chunk.page_content), "heuristic"),
                )
                graded.append(GradedChunk(doc=chunk, relevance=relevance, reason=reason))
            return graded
    except Exception:
        pass

    return [
        GradedChunk(doc=chunk, relevance=_heuristic_relevance(question, chunk.page_content), reason="heuristic")
        for chunk in selected
    ]
