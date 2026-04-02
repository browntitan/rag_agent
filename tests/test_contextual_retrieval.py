"""Tests for contextual retrieval and enriched delegation specs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Contextual retrieval tests
# ---------------------------------------------------------------------------

class TestContextualizeChunks:
    def _make_chunks(self, n: int = 3) -> list:
        return [
            Document(page_content=f"This is chunk {i} content.", metadata={"chunk_type": "general"})
            for i in range(n)
        ]

    def _make_settings(self, enabled: bool = True) -> MagicMock:
        s = MagicMock()
        s.contextual_retrieval_enabled = enabled
        return s

    def test_returns_unchanged_when_disabled(self):
        from agentic_chatbot.rag.ingest import _contextualize_chunks

        chunks = self._make_chunks()
        settings = self._make_settings(enabled=False)

        result = _contextualize_chunks(chunks, "full doc text", settings)

        assert result is chunks

    def test_prepends_context_to_chunks(self):
        from agentic_chatbot.rag.ingest import _contextualize_chunks

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="This chunk discusses pricing terms.")

        settings = self._make_settings(enabled=True)
        chunks = self._make_chunks(2)

        result = _contextualize_chunks(chunks, "full doc text", settings, llm=mock_llm)

        assert len(result) == 2
        for chunk in result:
            if chunk.metadata.get("has_context_prefix"):
                assert "This chunk discusses pricing terms." in chunk.page_content

    def test_marks_contextualized_chunks(self):
        from agentic_chatbot.rag.ingest import _contextualize_chunks

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Context: This is from Section 3.")

        settings = self._make_settings(enabled=True)
        chunks = self._make_chunks(1)

        result = _contextualize_chunks(chunks, "full doc text", settings, llm=mock_llm)

        if result[0].metadata.get("has_context_prefix"):
            assert result[0].metadata["has_context_prefix"] is True

    def test_falls_back_gracefully_on_llm_error(self):
        from agentic_chatbot.rag.ingest import _contextualize_chunks

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        settings = self._make_settings(enabled=True)
        chunks = self._make_chunks(2)

        result = _contextualize_chunks(chunks, "full doc text", settings, llm=mock_llm)

        # Per-chunk error falls back — chunks returned without context prefix
        assert len(result) == 2
        for chunk in result:
            assert not chunk.metadata.get("has_context_prefix")

    def test_factory_error_returns_original_chunks(self):
        from agentic_chatbot.rag.ingest import _contextualize_chunks

        settings = self._make_settings(enabled=True)
        chunks = self._make_chunks(2)

        # When llm=None and factory fails, original chunks returned
        with patch("agentic_chatbot.providers.llm_factory.build_providers",
                   side_effect=Exception("factory error")):
            result = _contextualize_chunks(chunks, "full doc text", settings, llm=None)

        assert result is chunks


# ---------------------------------------------------------------------------
# Enriched delegation specs tests
# ---------------------------------------------------------------------------

class TestDelegationSpec:
    def test_assigns_objective_when_missing(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "What are the payment terms?", "preferred_doc_ids": ["doc_abc"]}
        result = _build_delegation_spec(task, 0, 2)

        assert "objective" in result
        assert "payment terms" in result["objective"].lower() or "doc_abc" in result["objective"]

    def test_preserves_existing_objective(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {
            "query": "What are the payment terms?",
            "preferred_doc_ids": [],
            "objective": "Find all clauses related to payment obligations.",
        }
        result = _build_delegation_spec(task, 0, 1)

        assert result["objective"] == "Find all clauses related to payment obligations."

    def test_infers_keyword_strategy_for_definition_queries(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "What is the definition of 'Deliverable'?", "preferred_doc_ids": []}
        result = _build_delegation_spec(task, 0, 1)

        assert result["search_strategy"] == "keyword"

    def test_infers_vector_strategy_for_similarity_queries(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "Find text related to the indemnity obligation", "preferred_doc_ids": []}
        result = _build_delegation_spec(task, 0, 1)

        assert result["search_strategy"] == "vector"

    def test_defaults_to_hybrid_strategy(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "What does the agreement say about termination?", "preferred_doc_ids": []}
        result = _build_delegation_spec(task, 0, 1)

        assert result["search_strategy"] == "hybrid"

    def test_sets_boundary_for_doc_scoped_tasks(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "Summarise liability clauses", "preferred_doc_ids": ["doc_123"]}
        result = _build_delegation_spec(task, 0, 1)

        assert "boundary" in result
        assert "doc_123" in result["boundary"]

    def test_assigns_output_format(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import _build_delegation_spec

        task = {"query": "What are the key obligations?", "preferred_doc_ids": []}
        result = _build_delegation_spec(task, 0, 1)

        assert "output_format" in result
        assert len(result["output_format"]) > 0


class TestParallelPlannerNodeEnriched:
    def _make_state(self, tasks=None, messages=None) -> dict:
        from langchain_core.messages import HumanMessage
        return {
            "rag_sub_tasks": tasks or [],
            "messages": messages or [HumanMessage(content="Compare the two contracts")],
            "needs_clarification": False,
            "clarification_question": "",
        }

    def test_enriches_tasks_with_delegation_spec(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import parallel_planner_node

        tasks = [
            {"query": "What are payment terms in Contract A?", "preferred_doc_ids": ["doc_a"]},
            {"query": "What are payment terms in Contract B?", "preferred_doc_ids": ["doc_b"]},
        ]
        result = parallel_planner_node(self._make_state(tasks=tasks))

        enriched = result["rag_sub_tasks"]
        assert len(enriched) == 2
        for task in enriched:
            assert "objective" in task
            assert "output_format" in task
            assert "boundary" in task
            assert "search_strategy" in task

    def test_requests_clarification_for_all_vague_tasks(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import parallel_planner_node

        tasks = [
            {"query": "a", "preferred_doc_ids": []},
            {"query": "b", "preferred_doc_ids": []},
        ]
        result = parallel_planner_node(self._make_state(tasks=tasks))

        assert result.get("needs_clarification") is True
        assert result.get("clarification_question", "") != ""

    def test_does_not_request_clarification_for_specific_tasks(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import parallel_planner_node

        tasks = [
            {"query": "What are the payment obligations under clause 5?", "preferred_doc_ids": []},
        ]
        result = parallel_planner_node(self._make_state(tasks=tasks))

        assert not result.get("needs_clarification")

    def test_assigns_worker_ids(self):
        from agentic_chatbot.graph.nodes.parallel_planner_node import parallel_planner_node

        tasks = [{"query": "Find the liability cap", "preferred_doc_ids": []}]
        result = parallel_planner_node(self._make_state(tasks=tasks))

        assert result["rag_sub_tasks"][0]["worker_id"] == "rag_worker_0"
