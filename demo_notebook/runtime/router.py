from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str  # BASIC | AGENT
    confidence: float
    reasons: list[str]


_TOOL_HINTS = re.compile(
    r"\b(search|find|compare|diff|extract|requirement|clause|document|policy|cite|evidence|tool)\b",
    re.IGNORECASE,
)

_HIGH_COMPLEXITY = re.compile(
    r"\b(contract|compliance|risk|security|obligation|traceability)\b",
    re.IGNORECASE,
)



def route_message(user_text: str, *, force_agent: bool = False) -> RouteDecision:
    if force_agent:
        return RouteDecision(route="AGENT", confidence=1.0, reasons=["forced"])

    reasons: list[str] = []
    text = user_text.strip()

    if _TOOL_HINTS.search(text):
        reasons.append("tool_like_intent")
    if _HIGH_COMPLEXITY.search(text):
        reasons.append("domain_complexity")
    if len(text) > 350:
        reasons.append("long_input")

    if reasons:
        confidence = 0.92 if len(reasons) > 1 else 0.78
        return RouteDecision(route="AGENT", confidence=confidence, reasons=reasons)

    return RouteDecision(route="BASIC", confidence=0.85, reasons=["small_talk_or_simple_request"])
