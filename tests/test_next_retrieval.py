from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.persistence.postgres.chunks import ScoredChunk
from agentic_chatbot_next.rag.retrieval import grade_chunks, merge_dedupe, retrieve_candidates


def _scored_chunk(*, doc_id: str, chunk_id: str, title: str, score: float, method: str = "vector") -> ScoredChunk:
    return ScoredChunk(
        doc=Document(
            page_content=f"content from {title}",
            metadata={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "source_type": "kb",
            },
        ),
        score=score,
        method=method,
    )


def test_retrieve_candidates_expands_title_matches_for_architecture_queries():
    calls: list[tuple[str, str | None, str | None]] = []

    prompt_chunk = _scored_chunk(
        doc_id="doc-prompts",
        chunk_id="doc-prompts#chunk0001",
        title="TEST_QUERIES.md",
        score=0.91,
    )
    architecture_chunk = _scored_chunk(
        doc_id="doc-arch",
        chunk_id="doc-arch#chunk0001",
        title="ARCHITECTURE.md",
        score=0.55,
    )

    def vector_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("vector", doc_id_filter, collection_id_filter))
        if doc_id_filter == "doc-arch":
            return [architecture_chunk]
        return [prompt_chunk]

    def keyword_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("keyword", doc_id_filter, collection_id_filter))
        return []

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [
                {"doc_id": "doc-arch", "title": "ARCHITECTURE.md", "score": 0.93}
            ]
        ),
    )

    result = retrieve_candidates(
        stores,
        "key implementation details architecture documentation",
        tenant_id="tenant-123",
        preferred_doc_ids=["doc-prompts", "doc-arch"],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        collection_id_filter="default",
    )

    merged_doc_ids = {(chunk.doc.metadata or {}).get("doc_id") for chunk in result["merged"]}

    assert ("vector", None, "default") in calls
    assert ("keyword", None, "default") in calls
    assert ("vector", "doc-arch", None) in calls
    assert ("keyword", "doc-arch", None) in calls
    assert "doc-arch" in merged_doc_ids
    boosted_architecture = next(
        chunk for chunk in result["vector"] if (chunk.doc.metadata or {}).get("doc_id") == "doc-arch"
    )
    assert boosted_architecture.score > architecture_chunk.score


def test_merge_dedupe_returns_results_sorted_by_score_descending():
    low = _scored_chunk(doc_id="doc-low", chunk_id="doc-low#chunk0001", title="LOW.md", score=0.2)
    high = _scored_chunk(doc_id="doc-high", chunk_id="doc-high#chunk0001", title="HIGH.md", score=0.9)

    merged = merge_dedupe([low, high])

    assert [chunk.doc.metadata["doc_id"] for chunk in merged] == ["doc-high", "doc-low"]


def test_grade_chunks_preserves_architecture_docs_via_title_hint():
    architecture_doc = Document(
        page_content="Implementation details for the next runtime kernel and routing flow.",
        metadata={"chunk_id": "arch#chunk0001", "title": "ARCHITECTURE.md"},
    )
    prompt_doc = Document(
        page_content="Prompt catalog for demo questions.",
        metadata={"chunk_id": "prompt#chunk0001", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "arch#chunk0001", "relevance": 0, "reason": "missed"}, {"chunk_id": "prompt#chunk0001", "relevance": 1, "reason": "partial"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs? Cite your sources.",
        chunks=[architecture_doc, prompt_doc],
        callbacks=[],
    )

    by_id = {item.doc.metadata["chunk_id"]: item for item in graded}
    assert by_id["arch#chunk0001"].relevance >= 2
    assert by_id["arch#chunk0001"].reason == "title_hint"


def test_grade_chunks_demotes_prompt_catalog_question_echo():
    prompt_doc = Document(
        page_content="What are the key implementation details in the architecture docs? Cite your sources.",
        metadata={"chunk_id": "prompt#chunk0001", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "prompt#chunk0001", "relevance": 3, "reason": "exact match"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs? Cite your sources.",
        chunks=[prompt_doc],
        callbacks=[],
    )

    assert graded[0].relevance <= 1
    assert graded[0].reason == "question_echo"


def test_grade_chunks_demotes_meta_catalog_for_architecture_queries():
    prompt_doc = Document(
        page_content="Grouped prompt sets for basic chat, citations, architecture docs, and upload analysis.",
        metadata={"chunk_id": "prompt#chunk0002", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "prompt#chunk0002", "relevance": 3, "reason": "semantically related"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="key implementation details architecture documentation",
        chunks=[prompt_doc],
        callbacks=[],
    )

    assert graded[0].relevance <= 1
    assert graded[0].reason == "meta_catalog"
