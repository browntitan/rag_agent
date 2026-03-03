from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterDecision:
    route: str  # "BASIC" | "AGENT"
    confidence: float
    reasons: list[str]


_TOOL_VERBS = re.compile(
    r"\b(" +
    r"use\s+tool|call\s+tool|tool\s+call|run\s+tool|execute|calculate|compute|" +
    r"look\s+up|search|find|retrieve|query|open\s+file|summarize\s+this|" +
    r"upload|attached|attachment|document|pdf|spreadsheet|" +
    r"compare\s+and\s+recommend|step\s+by\s+step|first\s+do\s+.*\s+then\s+" +
    r")\b",
    flags=re.IGNORECASE,
)

_CITATION_HINTS = re.compile(r"\b(cite|citations|sources|evidence|grounded|according\s+to)\b", re.IGNORECASE)

_HIGH_STAKES_HINTS = re.compile(
    r"\b(medical|diagnosis|legal|contract|financial|tax|security\s+incident|compliance)\b",
    re.IGNORECASE,
)


def route_message(
    user_text: str,
    *,
    has_attachments: bool,
    explicit_force_agent: bool = False,
) -> RouterDecision:
    """Rule-based router.

    We keep the router deterministic and cheap. Escalate to AGENT whenever:
    - attachments are present
    - tool-like verbs are present
    - citations are requested
    - high-stakes keywords are present

    A separate safety net can escalate to AGENT if BASIC response is low confidence.
    """

    reasons: list[str] = []

    if explicit_force_agent:
        return RouterDecision(route="AGENT", confidence=1.0, reasons=["explicit_force_agent"])

    if has_attachments:
        return RouterDecision(route="AGENT", confidence=1.0, reasons=["attachments_present"])

    if _TOOL_VERBS.search(user_text):
        reasons.append("tool_or_multistep_intent")

    if _CITATION_HINTS.search(user_text):
        reasons.append("citation_or_grounding_requested")

    if _HIGH_STAKES_HINTS.search(user_text):
        reasons.append("high_stakes_topic")

    # Heuristic complexity: long messages tend to benefit from the agent.
    if len(user_text.strip()) > 600:
        reasons.append("long_input")

    if reasons:
        # Not all reasons are equal; attachments would have returned above.
        confidence = 0.75 if len(reasons) == 1 else 0.9
        return RouterDecision(route="AGENT", confidence=confidence, reasons=reasons)

    return RouterDecision(route="BASIC", confidence=0.85, reasons=["general_knowledge_or_small_talk"])
