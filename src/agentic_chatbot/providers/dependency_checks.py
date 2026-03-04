from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from agentic_chatbot.config import Settings


@dataclass(frozen=True)
class DependencyIssue:
    module: str
    contexts: Tuple[str, ...]
    hint: str


_MODULE_HINTS: Mapping[str, str] = {
    "langchain_ollama": "python -m pip install langchain-ollama",
    "langchain_openai": "python -m pip install langchain-openai",
}


def _required_module_map(settings: Settings) -> Dict[str, Set[str]]:
    required: Dict[str, Set[str]] = {}

    provider_by_context = {
        "llm": settings.llm_provider.lower(),
        "judge": settings.judge_provider.lower(),
        "embeddings": settings.embeddings_provider.lower(),
    }

    for context, provider in provider_by_context.items():
        if provider == "ollama":
            required.setdefault("langchain_ollama", set()).add(context)
        elif provider == "azure":
            required.setdefault("langchain_openai", set()).add(context)

    return required


def validate_provider_dependencies(settings: Settings) -> List[DependencyIssue]:
    issues: List[DependencyIssue] = []
    required = _required_module_map(settings)

    for module_name, contexts in sorted(required.items()):
        if importlib.util.find_spec(module_name) is None:
            issues.append(
                DependencyIssue(
                    module=module_name,
                    contexts=tuple(sorted(contexts)),
                    hint=_MODULE_HINTS.get(module_name, f"python -m pip install {module_name}"),
                )
            )

    return issues


def format_dependency_issues(issues: Iterable[DependencyIssue]) -> str:
    rows = list(issues)
    if not rows:
        return "No provider dependency issues detected."

    lines: List[str] = ["Missing provider package dependencies detected:"]
    hints: Set[str] = set()

    for issue in rows:
        contexts = ", ".join(issue.contexts)
        lines.append(f"- {issue.module} (required by: {contexts})")
        hints.add(issue.hint)

    lines.extend(
        [
            "",
            "Recommended fixes:",
            "1) Install all project dependencies: python -m pip install -r requirements.txt",
        ]
    )

    next_step = 2
    for hint in sorted(hints):
        lines.append(f"{next_step}) Install missing provider package: {hint}")
        next_step += 1

    lines.extend(
        [
            f"{next_step}) If running via Docker, rebuild and restart app: docker compose up -d --build app",
            f"{next_step + 1}) Re-run preflight: python run.py doctor",
        ]
    )
    return "\n".join(lines)


class ProviderDependencyError(RuntimeError):
    def __init__(self, issues: Iterable[DependencyIssue]):
        self.issues: Tuple[DependencyIssue, ...] = tuple(issues)
        super().__init__(format_dependency_issues(self.issues))


def raise_if_missing_provider_dependencies(settings: Settings) -> None:
    issues = validate_provider_dependencies(settings)
    if issues:
        raise ProviderDependencyError(issues)
