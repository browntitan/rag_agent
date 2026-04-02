"""Tests for the GraphRAG integration module."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGraphRAGConfig:
    def test_generates_project_directory(self, tmp_path):
        from agentic_chatbot.graphrag.config import generate_graphrag_settings

        settings = MagicMock()
        settings.graphrag_data_dir = tmp_path / "graphrag"
        settings.graphrag_completion_model = "gpt-4.1-mini"
        settings.graphrag_embedding_model = "text-embedding-3-small"
        settings.graphrag_chunk_size = 1200
        settings.graphrag_chunk_overlap = 100

        project_dir = generate_graphrag_settings(settings, "doc_test123")

        assert project_dir.exists()
        assert (project_dir / "input").exists()

    def test_writes_settings_file(self, tmp_path):
        from agentic_chatbot.graphrag.config import generate_graphrag_settings

        settings = MagicMock()
        settings.graphrag_data_dir = tmp_path / "graphrag"
        settings.graphrag_completion_model = "gpt-4.1-mini"
        settings.graphrag_embedding_model = "text-embedding-3-small"
        settings.graphrag_chunk_size = 1200
        settings.graphrag_chunk_overlap = 100

        project_dir = generate_graphrag_settings(settings, "doc_test456")

        # Should have either settings.yaml or settings.json
        has_settings = (project_dir / "settings.yaml").exists() or (project_dir / "settings.json").exists()
        assert has_settings

    def test_correct_doc_id_directory_name(self, tmp_path):
        from agentic_chatbot.graphrag.config import generate_graphrag_settings

        settings = MagicMock()
        settings.graphrag_data_dir = tmp_path / "graphrag"
        settings.graphrag_completion_model = "gpt-4.1-mini"
        settings.graphrag_embedding_model = "text-embedding-3-small"
        settings.graphrag_chunk_size = 1200
        settings.graphrag_chunk_overlap = 100

        project_dir = generate_graphrag_settings(settings, "my_special_doc")

        assert project_dir.name == "my_special_doc"


class TestGraphRAGIndexer:
    def test_returns_false_when_graphrag_not_installed(self, tmp_path):
        from agentic_chatbot.graphrag.indexer import run_graphrag_index

        with patch("shutil.which", return_value=None):
            result = run_graphrag_index(tmp_path)

        assert result is False

    def test_returns_true_on_success(self, tmp_path):
        from agentic_chatbot.graphrag.indexer import run_graphrag_index

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/graphrag"), \
             patch("subprocess.run", return_value=mock_result):
            result = run_graphrag_index(tmp_path)

        assert result is True

    def test_returns_false_on_nonzero_exit(self, tmp_path):
        from agentic_chatbot.graphrag.indexer import run_graphrag_index

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: something went wrong"

        with patch("shutil.which", return_value="/usr/bin/graphrag"), \
             patch("subprocess.run", return_value=mock_result):
            result = run_graphrag_index(tmp_path)

        assert result is False

    def test_returns_false_on_timeout(self, tmp_path):
        from agentic_chatbot.graphrag.indexer import run_graphrag_index

        with patch("shutil.which", return_value="/usr/bin/graphrag"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("graphrag", 600)):
            result = run_graphrag_index(tmp_path)

        assert result is False

    def test_graphrag_available_false_when_not_on_path(self):
        from agentic_chatbot.graphrag.indexer import graphrag_available

        with patch("shutil.which", return_value=None):
            assert graphrag_available() is False

    def test_graphrag_available_true_when_on_path(self):
        from agentic_chatbot.graphrag.indexer import graphrag_available

        with patch("shutil.which", return_value="/usr/local/bin/graphrag"):
            assert graphrag_available() is True


class TestGraphRAGSearcher:
    def test_returns_error_when_graphrag_not_installed(self, tmp_path):
        from agentic_chatbot.graphrag.searcher import graph_search

        with patch("shutil.which", return_value=None):
            result = graph_search("test query", tmp_path)

        assert "not installed" in result.lower()

    def test_returns_not_indexed_when_output_missing(self, tmp_path):
        from agentic_chatbot.graphrag.searcher import graph_search

        with patch("shutil.which", return_value="/usr/bin/graphrag"):
            result = graph_search("test query", tmp_path)

        assert "No GraphRAG index" in result or "index" in result.lower()

    def test_returns_search_result_on_success(self, tmp_path):
        from agentic_chatbot.graphrag.searcher import graph_search

        (tmp_path / "output").mkdir()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Company X is a large technology firm."

        with patch("shutil.which", return_value="/usr/bin/graphrag"), \
             patch("subprocess.run", return_value=mock_result):
            result = graph_search("Who is Company X?", tmp_path, method="local")

        assert result == "Company X is a large technology firm."

    def test_list_indexed_documents_empty_dir(self, tmp_path):
        from agentic_chatbot.graphrag.searcher import list_indexed_documents

        result = list_indexed_documents(tmp_path / "nonexistent")
        assert result == []

    def test_list_indexed_documents_finds_completed_indexes(self, tmp_path):
        from agentic_chatbot.graphrag.searcher import list_indexed_documents

        (tmp_path / "doc_abc" / "output").mkdir(parents=True)
        (tmp_path / "doc_xyz" / "output").mkdir(parents=True)
        (tmp_path / "doc_incomplete").mkdir()  # no output dir

        result = list_indexed_documents(tmp_path)
        assert set(result) == {"doc_abc", "doc_xyz"}
