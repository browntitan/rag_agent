"""Unit tests for extended RAG tools."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_chatbot.db.chunk_store import ChunkRecord


def _make_stores(chunk=None, chunks=None):
    """Return a mock KnowledgeStores."""
    stores = MagicMock()
    stores.chunk_store.get_chunk_by_id.return_value = chunk
    stores.chunk_store.get_chunks_by_index_range.return_value = chunks or []
    stores.chunk_store.vector_search.return_value = []
    return stores


def _make_session(tenant_id="local-dev"):
    s = MagicMock()
    s.tenant_id = tenant_id
    return s


def _make_settings(**kwargs):
    s = MagicMock()
    s.rag_top_k_vector = 8
    s.web_search_enabled = False
    s.tavily_api_key = None
    s.default_tenant_id = "local-dev"
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _make_judge(response_text=""):
    j = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    j.invoke.return_value = resp
    return j


def _chunk(chunk_id="doc#chunk0001", doc_id="doc", chunk_index=5, content="Test content."):
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=chunk_index,
        content=content,
        tenant_id="local-dev",
    )


# Import the factory once per test module to avoid repeated imports
def _get_tools(stores=None, session=None, judge=None, settings=None):
    from agentic_chatbot.tools.rag_tools_extended import make_extended_rag_tools
    stores = stores or _make_stores()
    session = session or _make_session()
    judge = judge or _make_judge()
    settings = settings or _make_settings()
    return make_extended_rag_tools(stores, session, judge_llm=judge, settings=settings)


class TestToolListIntegrity:
    def test_returns_five_tools(self):
        tools = _get_tools()
        assert len(tools) == 5

    def test_tool_names_are_unique(self):
        tools = _get_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_expected_tool_names_present(self):
        tools = _get_tools()
        names = {t.name for t in tools}
        assert "query_rewriter" in names
        assert "chunk_expander" in names
        assert "document_summarizer" in names
        assert "citation_validator" in names
        assert "web_search_fallback" in names


class TestQueryRewriter:
    def _get_tool(self, judge=None):
        tools = _get_tools(judge=judge)
        return next(t for t in tools if t.name == "query_rewriter")

    def test_returns_valid_json(self):
        judge = _make_judge()
        # rewrite_query returns original on failure; mock it to return a rewrite
        with patch("agentic_chatbot.rag.rewrite.rewrite_query", return_value="improved query") as m:
            tool = self._get_tool(judge)
            result = json.loads(tool.invoke({"query": "original query", "reason": "test"}))
        assert "original" in result
        assert "rewritten" in result
        assert "changed" in result
        assert result["rewritten"] == "improved query"
        assert result["changed"] is True

    def test_returns_original_on_failure(self):
        judge = _make_judge()
        with patch("agentic_chatbot.rag.rewrite.rewrite_query", side_effect=RuntimeError("boom")):
            tool = self._get_tool(judge)
            result = json.loads(tool.invoke({"query": "my query"}))
        assert result["original"] == "my query"
        assert result["changed"] is False


class TestChunkExpander:
    def _get_tool(self, stores):
        tools = _get_tools(stores=stores)
        return next(t for t in tools if t.name == "chunk_expander")

    def test_returns_neighbours(self):
        ch = _chunk(chunk_index=5)
        neighbours = [
            _chunk(chunk_id="doc#chunk0003", chunk_index=3),
            _chunk(chunk_id="doc#chunk0004", chunk_index=4),
            ch,
            _chunk(chunk_id="doc#chunk0006", chunk_index=6),
            _chunk(chunk_id="doc#chunk0007", chunk_index=7),
        ]
        stores = _make_stores(chunk=ch, chunks=neighbours)
        tool = self._get_tool(stores)
        result = json.loads(tool.invoke({"chunk_id": "doc#chunk0001", "window": 2}))
        assert result["count"] == 5
        assert len(result["chunks"]) == 5

    def test_chunk_not_found_returns_error(self):
        stores = _make_stores(chunk=None)
        tool = self._get_tool(stores)
        result = json.loads(tool.invoke({"chunk_id": "missing#chunk0001"}))
        assert "error" in result
        assert result["chunks"] == []

    def test_window_default_is_two(self):
        ch = _chunk(chunk_index=3)
        stores = _make_stores(chunk=ch, chunks=[ch])
        tool = self._get_tool(stores)
        tool.invoke({"chunk_id": "doc#chunk0001"})
        # Window=2 → get_chunks_by_index_range called with min=1, max=5
        stores.chunk_store.get_chunks_by_index_range.assert_called_once_with(
            ch.doc_id, 1, 5, "local-dev"
        )

    def test_window_clamped_at_zero_for_early_chunks(self):
        ch = _chunk(chunk_index=1)
        stores = _make_stores(chunk=ch, chunks=[ch])
        tool = self._get_tool(stores)
        tool.invoke({"chunk_id": "doc#chunk0001", "window": 5})
        # min_idx should clamp to 0 (max(0, 1-5) = 0)
        call_args = stores.chunk_store.get_chunks_by_index_range.call_args
        assert call_args[0][1] == 0  # min_idx argument


class TestDocumentSummarizer:
    def _get_tool(self, stores=None, judge=None):
        tools = _get_tools(stores=stores, judge=judge)
        return next(t for t in tools if t.name == "document_summarizer")

    def test_returns_summary_json(self):
        from langchain_core.documents import Document
        from agentic_chatbot.db.chunk_store import ScoredChunk

        # Create mock vector search results
        doc = Document(
            page_content="Contract clause 1 content.",
            metadata={"chunk_id": "doc#chunk0001", "chunk_index": 0},
        )
        scored = ScoredChunk(doc=doc, score=0.9, method="vector")
        stores = _make_stores()
        stores.chunk_store.vector_search.return_value = [scored]

        judge = _make_judge("This document covers payment terms and obligations.")
        tool = self._get_tool(stores=stores, judge=judge)
        result = json.loads(tool.invoke({"doc_id": "doc123", "focus": "payment"}))

        assert "summary" in result
        assert result["doc_id"] == "doc123"
        assert isinstance(result["chunk_ids_used"], list)

    def test_no_chunks_returns_error(self):
        stores = _make_stores()
        stores.chunk_store.vector_search.return_value = []
        tool = self._get_tool(stores=stores)
        result = json.loads(tool.invoke({"doc_id": "nonexistent"}))
        assert "error" in result
        assert result["summary"] == ""


class TestCitationValidator:
    def _get_tool(self, stores=None, judge=None):
        tools = _get_tools(stores=stores, judge=judge)
        return next(t for t in tools if t.name == "citation_validator")

    def test_supported_true(self):
        ch = _chunk(content="Monthly reports must be submitted by the 5th.")
        stores = _make_stores(chunk=ch)
        judge = _make_judge('{"supported": true, "confidence": 0.95, "reason": "Directly states it."}')
        tool = self._get_tool(stores=stores, judge=judge)
        result = json.loads(tool.invoke({
            "claim": "Monthly reports are required",
            "chunk_id": "doc#chunk0001",
        }))
        assert result["supported"] is True
        assert 0.0 <= result["confidence"] <= 1.0
        assert "reason" in result

    def test_chunk_not_found_returns_not_supported(self):
        stores = _make_stores(chunk=None)
        tool = self._get_tool(stores=stores)
        result = json.loads(tool.invoke({
            "claim": "Some claim",
            "chunk_id": "missing#chunk0001",
        }))
        assert result["supported"] is False
        assert result["confidence"] == 0.0

    def test_llm_failure_returns_safe_default(self):
        ch = _chunk()
        stores = _make_stores(chunk=ch)
        judge = MagicMock()
        judge.invoke.side_effect = RuntimeError("timeout")
        tool = self._get_tool(stores=stores, judge=judge)
        result = json.loads(tool.invoke({"claim": "test", "chunk_id": "doc#chunk0001"}))
        assert result["supported"] is False


class TestWebSearchFallback:
    def _get_tool(self, settings=None):
        tools = _get_tools(settings=settings)
        return next(t for t in tools if t.name == "web_search_fallback")

    def test_disabled_by_default(self):
        settings = _make_settings(web_search_enabled=False)
        tool = self._get_tool(settings=settings)
        result = json.loads(tool.invoke({"query": "test query"}))
        assert "error" in result
        assert result["results"] == []

    def test_no_api_key_returns_error(self):
        settings = _make_settings(web_search_enabled=True, tavily_api_key=None)
        tool = self._get_tool(settings=settings)
        result = json.loads(tool.invoke({"query": "test"}))
        assert "error" in result
        assert "TAVILY_API_KEY" in result["error"]

    def test_import_error_returns_graceful_message(self):
        settings = _make_settings(web_search_enabled=True, tavily_api_key="fake-key")
        tool = self._get_tool(settings=settings)
        with patch.dict("sys.modules", {"langchain_community": None,
                                         "langchain_community.tools": None,
                                         "langchain_community.tools.tavily_search": None}):
            result = json.loads(tool.invoke({"query": "test"}))
        assert result["results"] == []
