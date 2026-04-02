from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.runtime.kernel import RuntimeKernel


def _settings(tmp_path: Path, *, agents_dir: Path, skills_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=agents_dir,
        skills_dir=skills_dir,
        runtime_events_enabled=True,
        max_worker_concurrency=2,
    )


def _write_agent(path: Path, *, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_runtime_kernel_rejects_unknown_tool_reference(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    skills_dir = tmp_path / "skills"
    agents_dir.mkdir()
    skills_dir.mkdir()
    (skills_dir / "general_agent.md").write_text("general prompt", encoding="utf-8")
    _write_agent(
        agents_dir / "general.md",
        body="""---
name: general
mode: react
description: bad tool config
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["does_not_exist"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {}
---
bad
""",
    )

    with pytest.raises(ValueError, match="unknown tool"):
        RuntimeKernel(_settings(tmp_path, agents_dir=agents_dir, skills_dir=skills_dir), providers=None, stores=None)


def test_runtime_kernel_rejects_invalid_worker_and_memory_scope(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    skills_dir = tmp_path / "skills"
    agents_dir.mkdir()
    skills_dir.mkdir()
    (skills_dir / "general_agent.md").write_text("general prompt", encoding="utf-8")
    _write_agent(
        agents_dir / "general.md",
        body="""---
name: general
mode: react
description: bad worker config
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: ["missing_worker"]
preload_skill_packs: []
memory_scopes: ["invalid_scope"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {}
---
bad
""",
    )

    with pytest.raises(ValueError, match="unknown worker|invalid memory scope"):
        RuntimeKernel(_settings(tmp_path, agents_dir=agents_dir, skills_dir=skills_dir), providers=None, stores=None)
