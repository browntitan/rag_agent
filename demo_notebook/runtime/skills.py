from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import NotebookSettings


@dataclass(frozen=True)
class SkillProfile:
    enabled: bool
    active_files: List[str]
    prompts: Dict[str, str]


def load_skill_text(path: Path) -> str:
    """Load a skill markdown file, returning empty text when missing."""
    try:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def compose_prompt(
    base_prompt: str,
    shared_skill: str,
    role_skill: str,
    showcase_override: str | None = None,
) -> str:
    """Compose final system prompt from base + optional skill layers."""
    parts: List[str] = [base_prompt.strip()]

    if shared_skill.strip():
        parts.append("## Shared Skills\n" + shared_skill.strip())

    if role_skill.strip():
        parts.append("## Role Skills\n" + role_skill.strip())

    if showcase_override and showcase_override.strip():
        parts.append("## Showcase Override\n" + showcase_override.strip())

    return "\n\n".join(p for p in parts if p).strip()


def build_skill_profile(settings: NotebookSettings, base_prompts: Dict[str, str]) -> SkillProfile:
    """Return prompt overrides for dedicated skills showcase mode only."""
    if not (settings.skills_enabled and settings.skills_showcase_mode):
        return SkillProfile(enabled=False, active_files=[], prompts=dict(base_prompts))

    skills_dir = settings.skills_dir

    shared_path = skills_dir / "shared.md"
    supervisor_path = skills_dir / "supervisor.md"
    rag_path = skills_dir / "rag_agent.md"
    general_path = skills_dir / "general_agent.md"
    utility_path = skills_dir / "utility_agent.md"
    showcase_override_path = skills_dir / "skills_showcase_override.md"

    shared = load_skill_text(shared_path)
    supervisor = load_skill_text(supervisor_path)
    rag = load_skill_text(rag_path)
    general = load_skill_text(general_path)
    utility = load_skill_text(utility_path)
    showcase_override = load_skill_text(showcase_override_path)

    prompts = {
        "supervisor": compose_prompt(base_prompts["supervisor"], shared, supervisor, showcase_override),
        "rag": compose_prompt(base_prompts["rag"], shared, rag, showcase_override),
        "general": compose_prompt(base_prompts["general"], shared, general, showcase_override),
        "utility": compose_prompt(base_prompts["utility"], shared, utility, showcase_override),
        "synthesis": compose_prompt(base_prompts["synthesis"], shared, utility, showcase_override),
    }

    active_files = [
        str(p)
        for p in [
            shared_path,
            supervisor_path,
            rag_path,
            general_path,
            utility_path,
            showcase_override_path,
        ]
        if p.exists()
    ]

    return SkillProfile(enabled=True, active_files=active_files, prompts=prompts)
