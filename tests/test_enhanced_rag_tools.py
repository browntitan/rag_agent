"""Tests for enhanced RAG tools: full_text_search_document, search_by_metadata,
and Reciprocal Rank Fusion in merge_dedupe.
"""
from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest


def _build_tools_no_skills(stores, session, settings=None):
    """Build RAG tools. Pass settings=None to skip the skills-search tool entirely."""
    from agentic_chatbot.tools.rag_tools import make_all_rag_tools
    # When settings=None, make_all_rag_tools skips the skills-search block.
    return make_all_rag_tools(stores, session, settings=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scored_chunk(chunk_id: str, doc_id: str, score: float, method: str) -> Any:
    from langchain_core.documents import Document
    from agentic_chatbot.db.chunk_store import ScoredChunk

    doc = Document(
        page_content=f"Content of {chunk_id}",
        metadata={"chunk_id": chunk_id, "doc_id": doc_id},
    )
    return ScoredChunk(doc=doc, score=score, method=method)


def _make_chunk_record(chunk_id: str, doc_id: str, content: str) -> Any:
    from agentic_chatbot.db.chunk_store import ChunkRecord

    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        chunk_type="general",
    )


# ---------------------------------------------------------------------------
# RRF tests
# ---------------------------------------------------------------------------

class TestMergeDedupeRRF:
    def test_rrf_disabled_keeps_highest_score(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        c1 = _make_scored_chunk("c1", "doc1", 0.9, "vector")
        c1_kw = _make_scored_chunk("c1", "doc1", 0.5, "keyword")
        result = merge_dedupe([c1, c1_kw], use_rrf=False)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_rrf_enabled_fuses_scores(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        # c1 appears in both vector and keyword → higher RRF score
        c1_v = _make_scored_chunk("c1", "doc1", 0.9, "vector")
        c2_v = _make_scored_chunk("c2", "doc1", 0.8, "vector")
        c1_k = _make_scored_chunk("c1", "doc1", 0.7, "keyword")
        c3_k = _make_scored_chunk("c3", "doc1", 0.9, "keyword")

        result = merge_dedupe([c1_v, c2_v, c1_k, c3_k], use_rrf=True)

        # All 3 unique chunks should appear
        keys = {r.doc.metadata["chunk_id"] for r in result}
        assert keys == {"c1", "c2", "c3"}

    def test_rrf_cross_list_chunk_labeled_hybrid(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        c1_v = _make_scored_chunk("c1", "doc1", 0.9, "vector")
        c1_k = _make_scored_chunk("c1", "doc1", 0.7, "keyword")
        result = merge_dedupe([c1_v, c1_k], use_rrf=True)

        c1_result = next(r for r in result if r.doc.metadata["chunk_id"] == "c1")
        assert c1_result.method == "hybrid_rrf"

    def test_rrf_single_list_preserves_method(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        c1 = _make_scored_chunk("c1", "doc1", 0.9, "vector")
        c2 = _make_scored_chunk("c2", "doc1", 0.8, "vector")
        result = merge_dedupe([c1, c2], use_rrf=True)

        for r in result:
            assert r.method == "vector"

    def test_rrf_sorted_descending(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        chunks = [_make_scored_chunk(f"c{i}", "doc1", 1.0 / (i + 1), "vector") for i in range(5)]
        result = merge_dedupe(chunks, use_rrf=True)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_input(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        result = merge_dedupe([], use_rrf=True)
        assert result == []

    def test_rrf_k_constant_affects_scores(self):
        from agentic_chatbot.rag.retrieval import merge_dedupe

        c1_v = _make_scored_chunk("c1", "doc1", 0.9, "vector")
        c1_k = _make_scored_chunk("c1", "doc1", 0.7, "keyword")

        result_k60 = merge_dedupe([c1_v, c1_k], use_rrf=True, k_constant=60)
        result_k10 = merge_dedupe([c1_v, c1_k], use_rrf=True, k_constant=10)

        # Smaller k → larger individual scores
        assert result_k10[0].score > result_k60[0].score


# ---------------------------------------------------------------------------
# full_text_search_document tool tests
# ---------------------------------------------------------------------------

class TestFullTextSearchDocumentTool:
    def _make_tool(self, chunk_records):
        """Build the full_text_search_document tool with mocked stores."""
        from agentic_chatbot.rag.stores import KnowledgeStores

        stores = MagicMock(spec=KnowledgeStores)
        stores.chunk_store.full_text_search_document.return_value = chunk_records
        stores.doc_store = MagicMock()

        session = MagicMock()
        session.tenant_id = "test-tenant"
        session.scratchpad = {}

        settings = MagicMock()
        settings.rag_top_k_vector = 8
        settings.rag_top_k_keyword = 8
        settings.default_tenant_id = "test-tenant"

        # Patch skills search to avoid filesystem dependency
        with patch("agentic_chatbot.tools.rag_tools.make_all_rag_tools") as mock_make:
            from agentic_chatbot.tools.rag_tools import make_all_rag_tools
            tools = make_all_rag_tools(stores, session, settings=settings)

        return tools, stores

    def _build_tools_with_mocks(self, chunk_records=None, doc_list=None):
        """Build RAG tools with plain MagicMock stores (no skills search)."""
        stores = MagicMock()
        stores.chunk_store.full_text_search_document.return_value = chunk_records or []
        stores.doc_store.list_documents.return_value = doc_list or []

        session = MagicMock()
        session.tenant_id = "test-tenant"
        session.scratchpad = {}

        tools = _build_tools_no_skills(stores, session)
        return tools, stores

    def test_full_text_search_returns_full_content(self):
        records = [
            _make_chunk_record("c1", "doc1", "This is the FULL text of chunk one. " * 10),
            _make_chunk_record("c2", "doc1", "Another complete chunk with all words. " * 5),
        ]
        tools, _ = self._build_tools_with_mocks(chunk_records=records)

        tool_names = {t.name for t in tools}
        assert "full_text_search_document" in tool_names

        fts_tool = next(t for t in tools if t.name == "full_text_search_document")
        result_json = fts_tool.invoke({"doc_id": "doc1", "query": "chunk"})
        result = json.loads(result_json)

        assert len(result) == 2
        # Full content — not truncated to 500 chars
        assert "content" in result[0]
        assert len(result[0]["content"]) > 100

    def test_full_text_search_caps_max_results(self):
        tools, stores = self._build_tools_with_mocks()

        fts_tool = next(t for t in tools if t.name == "full_text_search_document")
        # max_results=200 should be capped to 50
        fts_tool.invoke({"doc_id": "doc1", "query": "test", "max_results": 200})
        # Verify the cap was applied
        call_kwargs = stores.chunk_store.full_text_search_document.call_args
        assert call_kwargs[1]["top_k"] == 50


# ---------------------------------------------------------------------------
# search_by_metadata tool tests
# ---------------------------------------------------------------------------

class TestSearchByMetadataTool:
    def _build_tools(self, doc_list):
        stores = MagicMock()
        stores.doc_store.list_documents.return_value = doc_list
        stores.chunk_store = MagicMock()

        session = MagicMock()
        session.tenant_id = "test-tenant"
        session.scratchpad = {}

        return _build_tools_no_skills(stores, session)

    def _make_doc(self, doc_id, title, source_type, file_type):
        d = MagicMock()
        d.doc_id = doc_id
        d.title = title
        d.source_type = source_type
        d.file_type = file_type
        d.num_chunks = 10
        d.ingested_at = "2024-01-01T00:00:00Z"
        return d

    def test_filter_by_source_type(self):
        docs = [
            self._make_doc("d1", "Contract A", "upload", "pdf"),
            self._make_doc("d2", "Policy Doc", "kb", "md"),
        ]
        tools = self._build_tools(docs)
        meta_tool = next(t for t in tools if t.name == "search_by_metadata")

        result = json.loads(meta_tool.invoke({"source_type": "upload"}))
        # list_documents is called with source_type filter; in-Python filter should match
        assert isinstance(result, list)

    def test_filter_by_title_contains(self):
        docs = [
            self._make_doc("d1", "Supply Chain Contract 2024", "kb", "pdf"),
            self._make_doc("d2", "HR Policy Document", "kb", "docx"),
        ]
        tools = self._build_tools(docs)
        meta_tool = next(t for t in tools if t.name == "search_by_metadata")

        result = json.loads(meta_tool.invoke({"title_contains": "Supply Chain"}))
        titles = [r["title"] for r in result]
        assert "Supply Chain Contract 2024" in titles
        assert "HR Policy Document" not in titles

    def test_empty_filters_return_all(self):
        docs = [
            self._make_doc("d1", "Doc A", "kb", "pdf"),
            self._make_doc("d2", "Doc B", "upload", "docx"),
        ]
        tools = self._build_tools(docs)
        meta_tool = next(t for t in tools if t.name == "search_by_metadata")

        result = json.loads(meta_tool.invoke({}))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tool list completeness
# ---------------------------------------------------------------------------

class TestToolListCompleteness:
    def test_all_14_tools_registered(self):
        stores = MagicMock()
        stores.doc_store.list_documents.return_value = []
        stores.chunk_store = MagicMock()

        session = MagicMock()
        session.tenant_id = "test-tenant"
        session.scratchpad = {}

        tools = _build_tools_no_skills(stores, session)

        expected = {
            "resolve_document",
            "search_document",
            "search_all_documents",
            "full_text_search_document",
            "search_by_metadata",
            "extract_clauses",
            "list_document_structure",
            "extract_requirements",
            "compare_clauses",
            "diff_documents",
            "scratchpad_write",
            "scratchpad_read",
            "scratchpad_list",
        }
        tool_names = {t.name for t in tools}
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"
