"""Unit tests for SkillsLoader — mtime cache, template substitution, validation."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agentic_chatbot.rag.skills_loader import SkillsLoader, _REQUIRED_SECTIONS


def _make_settings(tmp_path: Path) -> MagicMock:
    """Return a mock Settings that points all skill paths at tmp_path."""
    s = MagicMock()
    s.skills_backend = "local"
    s.shared_skills_path = tmp_path / "skills.md"
    s.general_agent_skills_path = tmp_path / "general_agent.md"
    s.rag_agent_skills_path = tmp_path / "rag_agent.md"
    s.supervisor_agent_skills_path = tmp_path / "supervisor_agent.md"
    s.utility_agent_skills_path = tmp_path / "utility_agent.md"
    s.basic_chat_skills_path = tmp_path / "basic_chat.md"
    return s


class TestSkillsLoaderDefaults:
    def test_returns_default_when_file_missing(self, tmp_path):
        s = _make_settings(tmp_path)
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        # Should contain the default RAG system text
        assert "Operating rules" in prompt

    def test_returns_default_when_file_empty(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("   ", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        assert "Operating rules" in prompt

    def test_all_six_agents_have_defaults(self, tmp_path):
        s = _make_settings(tmp_path)
        loader = SkillsLoader(s)
        for key in ["general_agent", "rag_agent", "supervisor_agent", "utility_agent", "basic_chat"]:
            result = loader.load(key)
            assert isinstance(result, str)
            assert len(result) > 10


class TestSkillsLoaderFileContent:
    def test_loads_file_content(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("# Custom RAG\nOperating rules: use tools.", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        assert "Custom RAG" in prompt

    def test_shared_preamble_prepended(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "skills.md").write_text("## Shared preamble", encoding="utf-8")
        (tmp_path / "rag_agent.md").write_text("## RAG specific\nOperating rules: x.", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        assert "Shared preamble" in prompt
        assert "RAG specific" in prompt
        assert "---" in prompt  # separator

    def test_no_shared_when_file_missing(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("## RAG specific\nOperating rules: x.", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        assert "---" not in prompt


class TestSkillsLoaderMtimeCache:
    def test_cache_hit_when_file_unchanged(self, tmp_path):
        s = _make_settings(tmp_path)
        path = tmp_path / "rag_agent.md"
        path.write_text("## RAG\nOperating rules: v1.", encoding="utf-8")
        loader = SkillsLoader(s)

        prompt1 = loader.load("rag_agent")
        # Overwrite file content but keep mtime unchanged by mocking stat
        path.write_text("## RAG\nOperating rules: v2.", encoding="utf-8")

        # Manually restore the mtime to what was cached
        old_mtime = loader._cache["rag_agent"].mtime
        import os
        os.utime(path, (old_mtime, old_mtime))

        prompt2 = loader.load("rag_agent")
        # Cache hit — should still return v1 content (mtime unchanged)
        assert prompt1 == prompt2

    def test_cache_miss_when_mtime_changes(self, tmp_path):
        s = _make_settings(tmp_path)
        path = tmp_path / "rag_agent.md"
        path.write_text("## RAG\nOperating rules: v1.", encoding="utf-8")
        loader = SkillsLoader(s)

        prompt1 = loader.load("rag_agent")
        assert "v1" in prompt1

        # Simulate file change: update content and bump mtime
        import time
        time.sleep(0.01)
        path.write_text("## RAG\nOperating rules: v2.", encoding="utf-8")

        prompt2 = loader.load("rag_agent")
        assert "v2" in prompt2
        assert prompt1 != prompt2

    def test_invalidate_clears_specific_key(self, tmp_path):
        s = _make_settings(tmp_path)
        path = tmp_path / "rag_agent.md"
        path.write_text("## RAG\nOperating rules: v1.", encoding="utf-8")
        loader = SkillsLoader(s)
        loader.load("rag_agent")
        assert "rag_agent" in loader._cache

        loader.invalidate("rag_agent")
        assert "rag_agent" not in loader._cache

    def test_invalidate_none_clears_all(self, tmp_path):
        s = _make_settings(tmp_path)
        for key in ["rag_agent", "supervisor_agent"]:
            fname = "rag_agent.md" if "rag" in key else "supervisor_agent.md"
            content = "Operating rules: x." if "rag" in key else "next_agent routing."
            (tmp_path / fname).write_text(content, encoding="utf-8")
        loader = SkillsLoader(s)
        loader.load("rag_agent")
        loader.load("supervisor_agent")
        assert len(loader._cache) == 2

        loader.invalidate()
        assert len(loader._cache) == 0


class TestSkillsLoaderTemplateSubstitution:
    def test_template_variable_substituted(self, tmp_path):
        s = _make_settings(tmp_path)
        content = "Operating rules: use {{tool_list}} to answer."
        (tmp_path / "rag_agent.md").write_text(content, encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent", context={"tool_list": "search_document, extract_clauses"})
        assert "search_document, extract_clauses" in prompt
        assert "{{tool_list}}" not in prompt

    def test_no_substitution_without_context(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("Use {{tool_list}}. Operating rules: x.", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent")
        assert "{{tool_list}}" in prompt  # not substituted

    def test_missing_variable_left_as_is(self, tmp_path):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("Tenant: {{tenant_name}}. Operating rules: x.", encoding="utf-8")
        loader = SkillsLoader(s)
        prompt = loader.load("rag_agent", context={"tool_list": "search"})
        assert "{{tenant_name}}" in prompt  # not in context, so left as-is


class TestSkillsLoaderValidation:
    def test_warning_for_missing_section(self, tmp_path, caplog):
        s = _make_settings(tmp_path)
        # rag_agent requires "Operating rules" — omit it
        (tmp_path / "rag_agent.md").write_text("# RAG Agent\nDo stuff.", encoding="utf-8")
        loader = SkillsLoader(s)
        with caplog.at_level(logging.WARNING):
            loader.load("rag_agent")
        assert any("Operating rules" in rec.message for rec in caplog.records)

    def test_no_warning_when_section_present(self, tmp_path, caplog):
        s = _make_settings(tmp_path)
        (tmp_path / "rag_agent.md").write_text("Operating rules:\n1. Use tools.", encoding="utf-8")
        loader = SkillsLoader(s)
        with caplog.at_level(logging.WARNING):
            loader.load("rag_agent")
        # No warning about missing sections
        section_warnings = [r for r in caplog.records if "missing expected section" in r.message]
        assert len(section_warnings) == 0
