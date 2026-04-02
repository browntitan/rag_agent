from __future__ import annotations

from pathlib import Path

import pytest

from agentic_chatbot_next.agents.loader import load_agent_markdown
from agentic_chatbot_next.agents.registry import AgentRegistry


def test_agent_registry_loads_markdown_definitions_from_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = AgentRegistry(repo_root / "data" / "agents")
    names = [agent.name for agent in registry.list()]
    assert names == sorted(names)
    assert set(names) >= {
        "basic",
        "general",
        "coordinator",
        "utility",
        "data_analyst",
        "rag_worker",
        "planner",
        "finalizer",
        "verifier",
        "memory_maintainer",
    }
    general = registry.get("general")
    assert general is not None
    assert general.prompt_file == "general_agent.md"
    assert "calculator" in general.allowed_tools


def test_repo_agents_directory_has_no_live_json_definitions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    json_files = sorted((repo_root / "data" / "agents").glob("*.json"))
    assert json_files == []


def test_agent_loader_rejects_missing_required_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text(
        "---\n"
        "name: broken\n"
        "mode: react\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt_file"):
        load_agent_markdown(path)
