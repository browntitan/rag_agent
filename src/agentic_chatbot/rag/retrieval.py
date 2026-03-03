from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from agentic_chatbot.db.chunk_store import ScoredChunk
from agentic_chatbot.rag.stores import KnowledgeStores

# Re-export ScoredChunk so callers that imported it from here continue to work.
__all__ = ["ScoredChunk", "vector_search", "keyword_search", "merge_dedupe", "retrieve_candidates"]


def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    return str(md.get("chunk_id") or f"{md.get('doc_id')}#{md.get('chunk_index')}")


def vector_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    doc_id_filter: Optional[str] = None,
) -> List[ScoredChunk]:
    """Vector similarity search via pgvector, with optional hard doc_id filter."""
    return stores.chunk_store.vector_search(query, top_k=top_k, doc_id_filter=doc_id_filter)


def keyword_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    doc_id_filter: Optional[str] = None,
) -> List[ScoredChunk]:
    """Full-text search via PostgreSQL tsvector, with optional hard doc_id filter."""
    return stores.chunk_store.keyword_search(query, top_k=top_k, doc_id_filter=doc_id_filter)


def merge_dedupe(chunks: Sequence[ScoredChunk]) -> List[ScoredChunk]:
    """Deduplicate by chunk_id, keeping the highest score per chunk."""
    by_key: Dict[str, ScoredChunk] = {}
    for c in chunks:
        k = _doc_key(c.doc)
        if k not in by_key or c.score > by_key[k].score:
            by_key[k] = c
    return list(by_key.values())


def retrieve_candidates(
    stores: KnowledgeStores,
    query: str,
    *,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    doc_id_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Fan-out vector + keyword retrieval.

    doc_id_filter:    Hard DB-level filter — only return chunks from this doc_id.
                      Takes precedence over preferred_doc_ids when set.
    preferred_doc_ids: Soft filter applied post-retrieval when doc_id_filter is None.
    must_include_uploads: Boost upload chunks slightly in score.
    """
    # Determine effective DB-level filter
    effective_filter: Optional[str] = doc_id_filter

    v = vector_search(stores, query, top_k=top_k_vector, doc_id_filter=effective_filter)
    k = keyword_search(stores, query, top_k=top_k_keyword, doc_id_filter=effective_filter)

    # Apply soft preferred_doc_ids filter when no hard filter is active
    if not effective_filter and preferred_doc_ids:
        v = [c for c in v if (c.doc.metadata or {}).get("doc_id") in preferred_doc_ids]
        k = [c for c in k if (c.doc.metadata or {}).get("doc_id") in preferred_doc_ids]

    # Boost upload chunks slightly
    if must_include_uploads:
        boosted_v: List[ScoredChunk] = []
        for c in v:
            if (c.doc.metadata or {}).get("source_type") == "upload":
                boosted_v.append(ScoredChunk(doc=c.doc, score=c.score + 0.01, method=c.method))
            else:
                boosted_v.append(c)
        v = boosted_v

    merged = merge_dedupe(v + k)

    return {
        "vector": v,
        "keyword": k,
        "merged": merged,
    }
