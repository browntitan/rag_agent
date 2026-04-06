from __future__ import annotations

from pathlib import Path


def test_readme_references_live_next_runtime_paths_and_supported_harness_assets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "src/agentic_chatbot_next/api/main.py" in readme
    assert "src/agentic_chatbot_next/cli.py" in readme
    assert "new_demo_notebook/README.md" in readme
    assert (repo_root / "new_demo_notebook" / "README.md").exists()
    assert (repo_root / "new_demo_notebook" / "demo_data" / "regional_spend.csv").exists()
