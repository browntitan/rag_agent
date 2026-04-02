from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_PREFIXES = (
    "agentic_chatbot.runtime",
    "agentic_chatbot.router",
    "agentic_chatbot.agents.orchestrator",
)

ALLOWED_SHARED_PREFIXES = (
    "agentic_chatbot.config",
    "agentic_chatbot.db",
    "agentic_chatbot.providers",
    "agentic_chatbot.sandbox.exceptions",
    "agentic_chatbot.rag.clause_splitter",
    "agentic_chatbot.rag.ocr",
    "agentic_chatbot.rag.structure_detector",
)


def test_next_runtime_does_not_import_legacy_runtime_packages() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    next_root = repo_root / "src" / "agentic_chatbot_next"
    violations: list[str] = []
    unexpected_shared_imports: list[str] = []

    for path in sorted(next_root.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(FORBIDDEN_PREFIXES):
                    violations.append(f"{path}: from {node.module} import ...")
                elif node.module.startswith("agentic_chatbot.") and not node.module.startswith(ALLOWED_SHARED_PREFIXES):
                    unexpected_shared_imports.append(f"{path}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_PREFIXES):
                        violations.append(f"{path}: import {alias.name}")
                    elif alias.name.startswith("agentic_chatbot.") and not alias.name.startswith(ALLOWED_SHARED_PREFIXES):
                        unexpected_shared_imports.append(f"{path}: import {alias.name}")

    assert not violations, "Forbidden legacy-runtime imports found:\n" + "\n".join(violations)
    assert not unexpected_shared_imports, (
        "Unexpected shared imports found. Next runtime may only depend on the documented shared layer:\n"
        + "\n".join(unexpected_shared_imports)
    )
