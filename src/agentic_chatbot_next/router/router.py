from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterDecision:
    route: str
    confidence: float
    reasons: list[str]
    suggested_agent: str = ""
    router_method: str = "deterministic"


_TOOL_VERBS = re.compile(
    r"\b("
    r"use\s+tool|call\s+tool|tool\s+call|run\s+tool|execute|calculate|compute|"
    r"look\s+up|search|find|retrieve|query|open\s+file|summarize\s+this|"
    r"upload|attached|attachment|document|pdf|spreadsheet|"
    r"compare|comparison|diff|difference|compare\s+and\s+recommend|step\s+by\s+step|first\s+do\s+.*\s+then\s+"
    r")\b",
    flags=re.IGNORECASE,
)

_DATA_ANALYSIS_HINTS = re.compile(
    r"\b("
    r"analyze\s+data|analyse\s+data|excel|csv|spreadsheet|dataframe|pandas|"
    r"average|mean|median|group\s+by|pivot|aggregate|filter\s+rows|"
    r"data\s+analysis|data\s+exploration|statistics|histogram|distribution|"
    r"sum\s+of|count\s+of|total\s+by|breakdown\s+by|top\s+\d+\s+by"
    r")\b",
    flags=re.IGNORECASE,
)

_CITATION_HINTS = re.compile(r"\b(cite|citations|sources|evidence|grounded|according\s+to)\b", re.IGNORECASE)

_HIGH_STAKES_HINTS = re.compile(
    r"\b(medical|diagnosis|legal|contract|financial|tax|security\s+incident|compliance)\b",
    re.IGNORECASE,
)

_COORDINATOR_HINTS = re.compile(
    r"\b("
    r"compare|comparison|difference|differences|across\s+documents|multi-step|step\s+by\s+step|"
    r"first\s+do\s+.*\s+then\s+|research|investigate|long[-\s]?running|background|parallel|"
    r"plan|orchestrate|coordinate|verify|synthesize"
    r")\b",
    re.IGNORECASE,
)


def route_message(
    user_text: str,
    *,
    has_attachments: bool,
    explicit_force_agent: bool = False,
) -> RouterDecision:
    reasons: list[str] = []

    if explicit_force_agent:
        return RouterDecision(route="AGENT", confidence=1.0, reasons=["explicit_force_agent"])

    if has_attachments:
        suggested = "coordinator" if _COORDINATOR_HINTS.search(user_text) else ""
        return RouterDecision(route="AGENT", confidence=1.0, reasons=["attachments_present"], suggested_agent=suggested)

    if _TOOL_VERBS.search(user_text):
        reasons.append("tool_or_multistep_intent")

    if _CITATION_HINTS.search(user_text):
        reasons.append("citation_or_grounding_requested")

    if _HIGH_STAKES_HINTS.search(user_text):
        reasons.append("high_stakes_topic")

    if _DATA_ANALYSIS_HINTS.search(user_text):
        reasons.append("data_analysis_intent")
        return RouterDecision(
            route="AGENT",
            confidence=0.90,
            reasons=reasons,
            suggested_agent="data_analyst",
        )

    suggested_agent = "coordinator" if _COORDINATOR_HINTS.search(user_text) else ""

    if len(user_text.strip()) > 600:
        reasons.append("long_input")

    if reasons:
        confidence = 0.75 if len(reasons) == 1 else 0.9
        return RouterDecision(route="AGENT", confidence=confidence, reasons=reasons, suggested_agent=suggested_agent)

    return RouterDecision(route="BASIC", confidence=0.85, reasons=["general_knowledge_or_small_talk"])
