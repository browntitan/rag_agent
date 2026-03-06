from __future__ import annotations

import ast
import sys
from pathlib import Path


FORBIDDEN_PREFIXES = ("agentic_chatbot",)


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "runtime"
    violations = []

    for py_file in sorted(root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_PREFIXES):
                        violations.append((py_file, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(FORBIDDEN_PREFIXES):
                    violations.append((py_file, module, node.lineno))

    if violations:
        print("Isolation check failed. Forbidden imports detected:")
        for path, module, lineno in violations:
            print(f"- {path}:{lineno} imports {module}")
        return 1

    print("Isolation check passed: no imports from production app modules were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
