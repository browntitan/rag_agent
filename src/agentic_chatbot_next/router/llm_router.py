from __future__ import annotations

import logging
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from agentic_chatbot_next.utils.json_utils import extract_json
from agentic_chatbot_next.router.router import RouterDecision, build_router_targets, route_message

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM_PROMPT = """\
You are a message router for an enterprise document-intelligence assistant.

## Your task
Classify the incoming user message as either BASIC or AGENT.

### Route to AGENT when the message:
- Asks about documents, contracts, policies, requirements, or procedures
- Requests search, retrieval, citations, or evidence
- Involves comparison or analysis of multiple documents
- Is a high-stakes domain (legal, medical, financial, compliance, security)
- Requires multi-step reasoning or tool use
- Contains file attachments or references to uploaded documents
- Involves data analysis, spreadsheets, Excel, CSV files, statistics, or pandas operations

### Route to BASIC when the message:
- Is a greeting or small talk
- Asks for general-knowledge information not tied to a specific document
- Is a simple conversational follow-up that was already answered

## Also suggest the best starting runtime agent
Choose from the runtime agents listed below, or return an empty string when the default top-level
agent is sufficient:
{agent_options}
"""

_ROUTER_HUMAN_TEMPLATE = """\
Recent conversation history (last 2 turns):
{history_summary}

Current user message:
{user_text}
"""


class LLMRouterOutput(BaseModel):
    route: str = Field(..., description="'BASIC' or 'AGENT'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence 0.0-1.0")
    reasoning: str = Field(..., description="One-sentence explanation of the routing decision")
    suggested_agent: str = Field(
        default="",
        description="Best starting runtime agent: 'coordinator' | 'data_analyst' | 'rag_worker' | ''",
    )

    @field_validator("route")
    @classmethod
    def _validate_route(cls, value: str) -> str:
        upper = value.strip().upper()
        if upper not in {"BASIC", "AGENT"}:
            raise ValueError(f"route must be 'BASIC' or 'AGENT', got {value!r}")
        return upper

    @field_validator("suggested_agent")
    @classmethod
    def _validate_suggested_agent(cls, value: str) -> str:
        return value.strip().lower()


def _agent_options_text(valid_suggested_agents: Iterable[str], descriptions: dict[str, str]) -> str:
    lines = ["- (empty string) - use the default top-level agent"]
    for agent_name in valid_suggested_agents:
        if not agent_name:
            continue
        description = descriptions.get(agent_name, "").strip() or "runtime specialist"
        lines.append(f"- {agent_name} - {description}")
    return "\n".join(lines)


def _sanitize_suggested_agent(value: str, valid_suggested_agents: Iterable[str]) -> str:
    valid = {str(item).strip().lower() for item in valid_suggested_agents if str(item).strip()}
    clean = str(value or "").strip().lower()
    return clean if clean in valid else ""


def route_turn(
    settings: object,
    providers: object,
    *,
    user_text: str,
    has_attachments: bool,
    history_summary: str = "",
    force_agent: bool = False,
    registry: Any | None = None,
):
    if bool(getattr(settings, "llm_router_enabled", True)):
        return route_message_hybrid(
            user_text,
            has_attachments=has_attachments,
            judge_llm=getattr(providers, "judge"),
            history_summary=history_summary,
            explicit_force_agent=force_agent,
            llm_confidence_threshold=float(getattr(settings, "llm_router_confidence_threshold", 0.70)),
            registry=registry,
        )
    return route_message(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=force_agent,
        registry=registry,
    )


__all__ = ["route_turn"]


def route_message_hybrid(
    user_text: str,
    *,
    has_attachments: bool,
    judge_llm: Any,
    history_summary: str = "",
    explicit_force_agent: bool = False,
    llm_confidence_threshold: float = 0.70,
    registry: Any | None = None,
) -> RouterDecision:
    targets = build_router_targets(registry)
    if explicit_force_agent:
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["explicit_force_agent"],
            suggested_agent=targets.rag_agent if any(token in user_text.lower() for token in ("cite", "source", "document", "architecture", "policy", "contract")) else "",
            router_method="deterministic",
        )

    if has_attachments:
        suggested = (
            targets.coordinator_agent
            if any(token in user_text.lower() for token in ("compare", "difference", "research", "analyze"))
            else (targets.rag_agent if any(token in user_text.lower() for token in ("cite", "source", "document", "architecture", "policy", "contract")) else "")
        )
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["attachments_present"],
            suggested_agent=suggested,
            router_method="deterministic",
        )

    deterministic = route_message(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=False,
        registry=registry,
    )
    if deterministic.confidence >= llm_confidence_threshold:
        return RouterDecision(
            route=deterministic.route,
            confidence=deterministic.confidence,
            reasons=deterministic.reasons,
            suggested_agent=deterministic.suggested_agent,
            router_method="deterministic",
        )

    logger.debug(
        "Deterministic confidence %.2f < threshold %.2f; escalating to LLM router.",
        deterministic.confidence,
        llm_confidence_threshold,
    )

    try:
        llm_out = _call_llm_router(
            judge_llm,
            user_text=user_text,
            history_summary=history_summary,
            valid_suggested_agents=targets.suggested_agents,
            descriptions=targets.descriptions,
        )
        return RouterDecision(
            route=llm_out.route,
            confidence=llm_out.confidence,
            reasons=[f"llm_router: {llm_out.reasoning}"],
            suggested_agent=llm_out.suggested_agent,
            router_method="llm",
        )
    except Exception as exc:
        logger.warning("LLM router failed (%s); falling back to deterministic route.", exc)
        return RouterDecision(
            route=deterministic.route,
            confidence=deterministic.confidence,
            reasons=deterministic.reasons + ["llm_router_failed"],
            suggested_agent=deterministic.suggested_agent,
            router_method="llm_fallback",
        )


def _call_llm_router(
    judge_llm: Any,
    *,
    user_text: str,
    history_summary: str,
    valid_suggested_agents: Iterable[str],
    descriptions: dict[str, str],
) -> LLMRouterOutput:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content=_ROUTER_SYSTEM_PROMPT.format(
                agent_options=_agent_options_text(valid_suggested_agents, descriptions),
            )
        ),
        HumanMessage(
            content=_ROUTER_HUMAN_TEMPLATE.format(
                history_summary=history_summary or "(no prior context)",
                user_text=user_text,
            )
        ),
    ]

    try:
        structured_llm = judge_llm.with_structured_output(LLMRouterOutput)
        result = structured_llm.invoke(messages)
        if isinstance(result, LLMRouterOutput):
            return result
    except (AttributeError, NotImplementedError, Exception):
        pass

    response = judge_llm.invoke(messages)
    text = getattr(response, "content", None) or str(response)
    return _parse_llm_response_text(text, valid_suggested_agents)


def _parse_llm_response_text(text: str, valid_suggested_agents: Iterable[str]) -> LLMRouterOutput:
    obj = extract_json(text) or {}
    route = str(obj.get("route", "")).strip().upper()
    if route not in {"BASIC", "AGENT"}:
        lower = text.lower()
        route = (
            "AGENT"
            if any(keyword in lower for keyword in ("agent", "document", "search", "rag", "retrieval"))
            else "BASIC"
        )

    confidence = float(obj.get("confidence", 0.65))
    confidence = max(0.0, min(1.0, confidence))

    suggested = _sanitize_suggested_agent(str(obj.get("suggested_agent", "")), valid_suggested_agents)

    reasoning = str(obj.get("reasoning", "parsed from text"))
    return LLMRouterOutput(
        route=route,
        confidence=confidence,
        reasoning=reasoning,
        suggested_agent=suggested,
    )
