"""LLM-based hybrid router for BASIC vs AGENT decision.

Strategy
--------
The deterministic router in ``router.py`` handles ~70 % of queries (greetings,
simple knowledge questions) with zero latency and cost.  This module adds a
*hybrid* variant that:

1. Runs the fast deterministic path first.
2. If the deterministic confidence is **below** ``llm_confidence_threshold``
   (default 0.70), escalates to a lightweight LLM call using the judge model.
3. If the LLM call fails for any reason, falls back to the deterministic result
   (so the system is never worse than before this file existed).

The LLM also returns a ``suggested_agent`` hint that the orchestrator can
use to choose a runtime-native starting agent when the request clearly
benefits from `coordinator` or `data_analyst`.

Usage::

    from agentic_chatbot.router import route_message_hybrid

    decision = route_message_hybrid(
        user_text,
        has_attachments=bool(upload_paths),
        judge_llm=providers.judge,
        history_summary=_summarise_history(session.messages, n=2),
        explicit_force_agent=force_agent,
        llm_confidence_threshold=settings.llm_router_confidence_threshold,
    )
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentic_chatbot.router.router import RouterDecision, route_message

logger = logging.getLogger(__name__)

_VALID_SUGGESTED_AGENTS = {"coordinator", "data_analyst", ""}

# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

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
- coordinator      — multi-step research, comparisons, background or long-running work, verification-heavy tasks
- data_analyst     — tabular data analysis (Excel, CSV), statistics, aggregations, pandas operations
- (empty string)   — general agent is sufficient
"""

_ROUTER_HUMAN_TEMPLATE = """\
Recent conversation history (last 2 turns):
{history_summary}

Current user message:
{user_text}
"""


class LLMRouterOutput(BaseModel):
    """Structured output expected from the judge LLM."""

    route: str = Field(..., description="'BASIC' or 'AGENT'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence 0.0–1.0")
    reasoning: str = Field(..., description="One-sentence explanation of the routing decision")
    suggested_agent: str = Field(
        default="",
        description=(
            "Best starting runtime agent: 'coordinator' | 'data_analyst' | ''"
        ),
    )

    @field_validator("route")
    @classmethod
    def _validate_route(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in {"BASIC", "AGENT"}:
            raise ValueError(f"route must be 'BASIC' or 'AGENT', got {v!r}")
        return upper

    @field_validator("suggested_agent")
    @classmethod
    def _validate_suggested_agent(cls, v: str) -> str:
        clean = v.strip().lower()
        return clean if clean in _VALID_SUGGESTED_AGENTS else ""


# ---------------------------------------------------------------------------
# Hybrid routing function
# ---------------------------------------------------------------------------

def route_message_hybrid(
    user_text: str,
    *,
    has_attachments: bool,
    judge_llm: Any,
    history_summary: str = "",
    explicit_force_agent: bool = False,
    llm_confidence_threshold: float = 0.70,
) -> RouterDecision:
    """Hybrid BASIC / AGENT router.

    Uses the deterministic router as a fast path.  Escalates to the judge LLM
    only when the deterministic confidence falls below *llm_confidence_threshold*.

    Args:
        user_text:                The current user message.
        has_attachments:          Whether file uploads accompany this turn.
        judge_llm:                A LangChain chat model (cheap/fast, e.g. the
                                  judge LLM from ``ProviderBundle``).
        history_summary:          Short textual summary of the last 2 turns,
                                  provided by the caller.
        explicit_force_agent:     If True, always route to AGENT regardless.
        llm_confidence_threshold: Deterministic confidence below which the LLM
                                  is consulted (default 0.70).

    Returns:
        RouterDecision with ``router_method`` set to one of:
        - "deterministic"  — fast path used, no LLM call made
        - "llm"            — LLM was consulted and succeeded
        - "llm_fallback"   — LLM call failed; deterministic result returned
    """
    # ── Fast path: clear-cut cases that never need LLM ──────────────────
    if explicit_force_agent:
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["explicit_force_agent"],
            suggested_agent="",
            router_method="deterministic",
        )

    if has_attachments:
        suggested = "coordinator" if any(token in user_text.lower() for token in ("compare", "difference", "research", "analyze")) else ""
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["attachments_present"],
            suggested_agent=suggested,
            router_method="deterministic",
        )

    # ── Deterministic route ──────────────────────────────────────────────
    det = route_message(user_text, has_attachments=has_attachments, explicit_force_agent=False)

    if det.confidence >= llm_confidence_threshold:
        # Confident enough — skip the LLM call entirely
        return RouterDecision(
            route=det.route,
            confidence=det.confidence,
            reasons=det.reasons,
            suggested_agent=det.suggested_agent,
            router_method="deterministic",
        )

    # ── LLM escalation ──────────────────────────────────────────────────
    logger.debug(
        "Deterministic confidence %.2f < threshold %.2f — escalating to LLM router",
        det.confidence,
        llm_confidence_threshold,
    )

    try:
        llm_out = _call_llm_router(judge_llm, user_text=user_text, history_summary=history_summary)
        return RouterDecision(
            route=llm_out.route,
            confidence=llm_out.confidence,
            reasons=[f"llm_router: {llm_out.reasoning}"],
            suggested_agent=llm_out.suggested_agent,
            router_method="llm",
        )
    except Exception as exc:
        logger.warning(
            "LLM router call failed (%s); falling back to deterministic result", exc
        )
        return RouterDecision(
            route=det.route,
            confidence=det.confidence,
            reasons=det.reasons + ["llm_router_failed"],
            suggested_agent=det.suggested_agent,
            router_method="llm_fallback",
        )


# ---------------------------------------------------------------------------
# Internal: invoke the judge LLM with structured output
# ---------------------------------------------------------------------------

def _call_llm_router(
    judge_llm: Any,
    *,
    user_text: str,
    history_summary: str,
) -> LLMRouterOutput:
    """Invoke the judge LLM and parse its structured routing decision.

    Uses ``with_structured_output`` when available (Azure, OpenAI-compatible),
    otherwise falls back to plain text + manual JSON extraction.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    human_content = _ROUTER_HUMAN_TEMPLATE.format(
        history_summary=history_summary or "(no prior context)",
        user_text=user_text,
    )
    messages = [
        SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    # Prefer structured output (type-safe, provider-native JSON mode)
    try:
        structured_llm = judge_llm.with_structured_output(LLMRouterOutput)
        result = structured_llm.invoke(messages)
        if isinstance(result, LLMRouterOutput):
            return result
    except (AttributeError, NotImplementedError, Exception):
        pass  # Fall through to text-based extraction below

    # Fallback: plain invoke + JSON extraction
    resp = judge_llm.invoke(messages)
    text = getattr(resp, "content", None) or str(resp)
    return _parse_llm_response_text(text)


def _parse_llm_response_text(text: str) -> LLMRouterOutput:
    """Extract an LLMRouterOutput from a free-text LLM response."""
    from agentic_chatbot.utils.json_utils import extract_json

    obj = extract_json(text) or {}
    route = str(obj.get("route", "")).strip().upper()
    if route not in {"BASIC", "AGENT"}:
        # Keyword heuristic if JSON is missing the route field
        lower = text.lower()
        route = "AGENT" if any(
            kw in lower for kw in ("agent", "document", "search", "rag", "retrieval")
        ) else "BASIC"

    confidence = float(obj.get("confidence", 0.65))
    confidence = max(0.0, min(1.0, confidence))

    suggested = str(obj.get("suggested_agent", "")).strip().lower()
    if suggested not in _VALID_SUGGESTED_AGENTS:
        suggested = ""

    reasoning = str(obj.get("reasoning", "parsed from text"))

    return LLMRouterOutput(
        route=route,
        confidence=confidence,
        reasoning=reasoning,
        suggested_agent=suggested,
    )
