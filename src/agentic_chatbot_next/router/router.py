from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouterDecision:
    route: str
    confidence: float
    reasons: list[str]
    suggested_agent: str = ""
    router_method: str = "deterministic"


@dataclass(frozen=True)
class RouterTargets:
    default_agent: str = "general"
    basic_agent: str = "basic"
    coordinator_agent: str = "coordinator"
    data_analyst_agent: str = "data_analyst"
    rag_agent: str = "rag_worker"
    suggested_agents: tuple[str, ...] = ("coordinator", "data_analyst", "rag_worker")
    descriptions: dict[str, str] = field(default_factory=dict)


def build_router_targets(registry: Any | None = None) -> RouterTargets:
    if registry is None:
        return RouterTargets(
            descriptions={
                "coordinator": "Manager-only role for explicit worker orchestration.",
                "data_analyst": "Tabular data analysis specialist using sandboxed Python tools.",
                "rag_worker": "Grounded document worker that returns the stable RAG contract.",
            }
        )

    default_agent = getattr(registry, "get_default_agent_name", lambda: "general")()
    basic_agent = getattr(registry, "get_basic_agent_name", lambda: "basic")()
    coordinator_agent = getattr(registry, "get_manager_agent_name", lambda: "coordinator")()
    data_analyst_agent = getattr(registry, "get_data_analyst_agent_name", lambda: "data_analyst")()
    rag_agent = getattr(registry, "get_rag_agent_name", lambda: "rag_worker")()

    descriptions: dict[str, str] = {}
    suggested: list[str] = []
    list_routable = getattr(registry, "list_routable", lambda: [])
    for agent in list_routable():
        if agent.mode == "basic":
            continue
        descriptions[agent.name] = agent.description
        if agent.name not in suggested:
            suggested.append(agent.name)
    for name in (coordinator_agent, data_analyst_agent, rag_agent, default_agent):
        if name and name != basic_agent and name not in suggested:
            suggested.append(name)
    return RouterTargets(
        default_agent=default_agent or "general",
        basic_agent=basic_agent or "basic",
        coordinator_agent=coordinator_agent or "coordinator",
        data_analyst_agent=data_analyst_agent or "data_analyst",
        rag_agent=rag_agent or "rag_worker",
        suggested_agents=tuple(suggested),
        descriptions=descriptions,
    )


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

_RAG_HINTS = re.compile(
    r"\b("
    r"cite|citations|sources|source|evidence|grounded|architecture|document|docs|kb|"
    r"policy|agreement|contract|requirements|playbook|runbook"
    r")\b",
    re.IGNORECASE,
)


def route_message(
    user_text: str,
    *,
    has_attachments: bool,
    explicit_force_agent: bool = False,
    registry: Any | None = None,
) -> RouterDecision:
    targets = build_router_targets(registry)
    reasons: list[str] = []

    if explicit_force_agent:
        return RouterDecision(route="AGENT", confidence=1.0, reasons=["explicit_force_agent"])

    if has_attachments:
        suggested = (
            targets.coordinator_agent
            if _COORDINATOR_HINTS.search(user_text)
            else (targets.rag_agent if _RAG_HINTS.search(user_text) else "")
        )
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
            suggested_agent=targets.data_analyst_agent,
        )

    if _COORDINATOR_HINTS.search(user_text):
        suggested_agent = targets.coordinator_agent
    elif _RAG_HINTS.search(user_text):
        suggested_agent = targets.rag_agent
    else:
        suggested_agent = ""

    if len(user_text.strip()) > 600:
        reasons.append("long_input")

    if reasons:
        confidence = 0.75 if len(reasons) == 1 else 0.9
        return RouterDecision(route="AGENT", confidence=confidence, reasons=reasons, suggested_agent=suggested_agent)

    return RouterDecision(route="BASIC", confidence=0.85, reasons=["general_knowledge_or_small_talk"])
