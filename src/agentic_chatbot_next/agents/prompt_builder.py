from __future__ import annotations

from pathlib import Path


class PromptBuilder:
    """Minimal prompt resolver for markdown-backed agent prompts."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir

    def load_prompt(self, prompt_file: str) -> str:
        path = self.skills_dir / prompt_file
        if not path.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file!r} was not found under {self.skills_dir}.")
        return path.read_text(encoding="utf-8")

