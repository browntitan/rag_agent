from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from model output."""

    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
