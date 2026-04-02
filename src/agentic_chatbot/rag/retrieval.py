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
    tenant_id: str,
    doc_id_filter: Optional[str] = None,
) -> List[ScoredChunk]:
    """Vector similarity search via pgvector, with optional hard doc_id filter."""
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
    """Full-text search via PostgreSQL tsvector, with optional hard doc_id filter."""
    return stores.chunk_store.keyword_search(
        query,
        top_k=top_k,
        doc_id_filter=doc_id_filter,
        tenant_id=tenant_id,
    )


def merge_dedupe(
    chunks: Sequence[ScoredChunk],
    *,
    use_rrf: bool = True,
    k_constant: int = 60,
) -> List[ScoredChunk]:
    """Deduplicate and optionally re-rank results using Reciprocal Rank Fusion (RRF).

    When ``use_rrf=True`` (default), chunks from separate ranked lists are fused
    using the RRF formula: ``score = sum(1 / (k + rank + 1))`` across all lists
    where ``rank`` is the 0-based position in that list.  Chunks appearing in
    both vector and keyword lists receive higher fused scores, naturally
    rewarding evidence that is both semantically and lexically relevant.

    When ``use_rrf=False``, falls back to simple dedup keeping the highest
    individual score per chunk — compatible with prior behaviour.

    Args:
        chunks:      Sequence of ScoredChunks from one or more retrieval methods.
                     Each chunk's ``.method`` attribute (``'vector'`` or ``'keyword'``)
                     is used to group into separate ranked lists before fusion.
        use_rrf:     Whether to apply RRF scoring (default: True).
        k_constant:  RRF smoothing constant (default: 60, standard literature value).

    Returns:
        Deduplicated list of ScoredChunks ordered by descending fused score.
        Chunks present in both lists are labelled ``method='hybrid_rrf'``.
    """
    if not use_rrf:
        # Legacy behaviour: keep highest-score-per-chunk
        by_key: Dict[str, ScoredChunk] = {}
        for c in chunks:
            k = _doc_key(c.doc)
            if k not in by_key or c.score > by_key[k].score:
                by_key[k] = c
        return list(by_key.values())

    # ── RRF fusion ────────────────────────────────────────────────────────
    # 1. Separate into ranked lists by retrieval method
    lists_by_method: Dict[str, List[ScoredChunk]] = {}
    for c in chunks:
        method = c.method or "unknown"
        lists_by_method.setdefault(method, []).append(c)

    # Each list is already ordered by its native score (descending).
    # Sort to be sure.
    for method_list in lists_by_method.values():
        method_list.sort(key=lambda x: x.score, reverse=True)

    # 2. Accumulate RRF scores and track methods per chunk
    rrf_scores: Dict[str, float] = {}
    rrf_methods: Dict[str, set] = {}
    rrf_chunk_ref: Dict[str, ScoredChunk] = {}

    for ranked_list in lists_by_method.values():
        for rank, chunk in enumerate(ranked_list):
            key = _doc_key(chunk.doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_constant + rank + 1)
            rrf_methods.setdefault(key, set()).add(chunk.method)
            if key not in rrf_chunk_ref:
                rrf_chunk_ref[key] = chunk

    # 3. Build output with updated scores and method labels
    fused: List[ScoredChunk] = []
    for key, fused_score in rrf_scores.items():
        ref = rrf_chunk_ref[key]
        methods_seen = rrf_methods[key]
        method_label = "hybrid_rrf" if len(methods_seen) > 1 else next(iter(methods_seen))
        fused.append(ScoredChunk(doc=ref.doc, score=round(fused_score, 6), method=method_label))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


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
    """Fan-out vector + keyword retrieval.

    doc_id_filter:    Hard DB-level filter — only return chunks from this doc_id.
                      Takes precedence over preferred_doc_ids when set.
    preferred_doc_ids: Soft filter applied post-retrieval when doc_id_filter is None.
    must_include_uploads: Boost upload chunks slightly in score.
    """
    # Determine effective DB-level filter
    effective_filter: Optional[str] = doc_id_filter

    v = vector_search(
        stores,
        query,
        top_k=top_k_vector,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
    )
    k = keyword_search(
        stores,
        query,
        top_k=top_k_keyword,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
    )

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
