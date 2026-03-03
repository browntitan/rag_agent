from __future__ import annotations

import ast
import math
from typing import Any, Dict

from langchain.tools import tool


_ALLOWED_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Load,
    ast.Name,
)


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("Only simple math functions are allowed")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"Unknown name: {node.id}")
    compiled = compile(tree, "<expr>", "eval")
    return float(eval(compiled, {"__builtins__": {}}, _ALLOWED_NAMES))


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely.

    Supports +, -, *, /, **, %, parentheses and a small set of functions:
    sqrt, log, log10, sin, cos, tan, floor, ceil, abs, round.

    Example:
        calculator("(12 * 40_000) / 1_000_000")
    """

    try:
        result = _safe_eval(expression)
        # Render integers without trailing .0
        if abs(result - int(result)) < 1e-12:
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"
