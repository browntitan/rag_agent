from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot.db.skill_store import SkillChunkMatch
from agentic_chatbot.rag.skill_index import load_skill_pack_from_file
from agentic_chatbot.tools.skills_search_tool import make_skills_search_tool


def test_load_skill_pack_from_file_parses_metadata_and_chunks(tmp_path: Path):
    skill_file = tmp_path / "comparison.md"
    skill_file.write_text(
        "# Comparison Workflow\n"
        "agent_scope: rag\n"
        "tool_tags: diff_documents, compare_clauses\n"
        "task_tags: comparison, diff\n"
        "version: 2\n"
        "enabled: true\n"
        "description: Compare documents carefully.\n\n"
        "## Workflow\n"
        "Analyze each document independently before synthesizing.\n"
    )

    pack = load_skill_pack_from_file(skill_file, root=tmp_path)

    assert pack.name == "Comparison Workflow"
    assert pack.agent_scope == "rag"
    assert pack.tool_tags == ["diff_documents", "compare_clauses"]
    assert pack.task_tags == ["comparison", "diff"]
    assert pack.version == "2"
    assert pack.enabled is True
    assert pack.chunks


class _FakeSkillStore:
    def vector_search(self, query, *, tenant_id, top_k, agent_scope, tool_tags=None, task_tags=None, enabled_only=True):
        return [
            SkillChunkMatch(
                skill_id="rag-comparison",
                name="Comparison Workflow",
                agent_scope=agent_scope or "rag",
                content="Use diff_documents before compare_clauses.",
                chunk_index=0,
                score=0.95,
                tool_tags=["diff_documents"],
                task_tags=["comparison"],
            )
        ]


def test_search_skills_prefers_db_backed_matches():
    settings = SimpleNamespace(
        default_tenant_id="local-dev",
        skill_search_top_k=4,
        skill_context_max_chars=4000,
    )
    stores = SimpleNamespace(skill_store=_FakeSkillStore())
    tool = make_skills_search_tool(settings, stores=stores)

    result = tool.invoke({"query": "how to compare two documents", "agent_filter": "rag_agent", "top_k": 1})

    assert "Comparison Workflow" in result
    assert "diff_documents" in result


def test_repo_skill_packs_cover_runtime_agent_scopes():
    root = Path(__file__).resolve().parents[1] / "data" / "skill_packs"
    scopes = {
        load_skill_pack_from_file(path, root=root).agent_scope
        for path in root.rglob("*.md")
    }

    assert {"rag", "general", "utility", "data_analyst", "planner", "finalizer", "verifier"} <= scopes
