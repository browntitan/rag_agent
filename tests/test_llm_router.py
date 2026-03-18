"""Unit tests for the LLM hybrid router."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_chatbot.router.router import RouterDecision, route_message
from agentic_chatbot.router.llm_router import (
    LLMRouterOutput,
    _call_llm_router,
    _parse_llm_response_text,
    route_message_hybrid,
)


# ---------------------------------------------------------------------------
# RouterDecision backward compatibility
# ---------------------------------------------------------------------------

class TestRouterDecisionBackwardsCompat:
    def test_existing_positional_args_still_work(self):
        d = RouterDecision(route="BASIC", confidence=0.85, reasons=["general_knowledge"])
        assert d.route == "BASIC"
        assert d.confidence == 0.85
        assert d.reasons == ["general_knowledge"]
        # New fields have defaults
        assert d.suggested_agent == ""
        assert d.router_method == "deterministic"

    def test_new_fields_can_be_set(self):
        d = RouterDecision(
            route="AGENT",
            confidence=0.9,
            reasons=["tool_intent"],
            suggested_agent="rag_agent",
            router_method="llm",
        )
        assert d.suggested_agent == "rag_agent"
        assert d.router_method == "llm"


# ---------------------------------------------------------------------------
# Fast paths (no LLM call)
# ---------------------------------------------------------------------------

class TestHybridRouterFastPaths:
    def _make_judge(self):
        return MagicMock()

    def test_explicit_force_agent_skips_llm(self):
        judge = self._make_judge()
        decision = route_message_hybrid(
            "hello",
            has_attachments=False,
            judge_llm=judge,
            explicit_force_agent=True,
        )
        assert decision.route == "AGENT"
        assert decision.confidence == 1.0
        assert decision.router_method == "deterministic"
        judge.invoke.assert_not_called()

    def test_has_attachments_skips_llm(self):
        judge = self._make_judge()
        decision = route_message_hybrid(
            "please review this",
            has_attachments=True,
            judge_llm=judge,
        )
        assert decision.route == "AGENT"
        assert decision.router_method == "deterministic"
        assert decision.suggested_agent == "rag_agent"
        judge.invoke.assert_not_called()

    def test_high_confidence_deterministic_skips_llm(self):
        """Deterministic router returns 0.85 for BASIC → should skip LLM."""
        judge = self._make_judge()
        decision = route_message_hybrid(
            "hi there",
            has_attachments=False,
            judge_llm=judge,
            llm_confidence_threshold=0.70,
        )
        # "hi there" should deterministically route to BASIC with confidence 0.85
        assert decision.route == "BASIC"
        assert decision.router_method == "deterministic"
        judge.with_structured_output.assert_not_called()


# ---------------------------------------------------------------------------
# LLM escalation path
# ---------------------------------------------------------------------------

class TestHybridRouterLLMEscalation:
    def _make_judge_with_structured_output(self, route, confidence, suggested_agent=""):
        """Return a mock judge whose with_structured_output().invoke() returns the given values."""
        llm_out = LLMRouterOutput(
            route=route,
            confidence=confidence,
            reasoning="test reasoning",
            suggested_agent=suggested_agent,
        )
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = llm_out

        judge = MagicMock()
        judge.with_structured_output.return_value = structured_mock
        return judge

    def test_llm_called_when_confidence_below_threshold(self):
        """Single-reason AGENT (confidence 0.75) should escalate to LLM.

        'calculate something' matches only _TOOL_VERBS → 1 reason → confidence 0.75.
        With threshold 0.76, the deterministic result falls below it, so LLM is consulted.
        """
        judge = self._make_judge_with_structured_output("AGENT", 0.9, "rag_agent")

        decision = route_message_hybrid(
            "calculate something",
            has_attachments=False,
            judge_llm=judge,
            llm_confidence_threshold=0.76,  # 0.75 deterministic < 0.76 → LLM escalation
        )
        assert decision.route == "AGENT"
        assert decision.router_method == "llm"
        assert decision.suggested_agent == "rag_agent"

    def test_llm_output_route_overrides_deterministic(self):
        """If LLM says BASIC when deterministic would say AGENT, LLM wins."""
        judge = self._make_judge_with_structured_output("BASIC", 0.80)

        decision = route_message_hybrid(
            "find something",
            has_attachments=False,
            judge_llm=judge,
            llm_confidence_threshold=0.80,
        )
        assert decision.route == "BASIC"
        assert decision.router_method == "llm"

    def test_suggested_agent_invalid_value_coerced_to_empty(self):
        """LLM returning an unknown agent name should be silently cleared."""
        llm_out = LLMRouterOutput(
            route="AGENT",
            confidence=0.9,
            reasoning="...",
            suggested_agent="unknown_agent",  # validator should coerce to ""
        )
        assert llm_out.suggested_agent == ""

    def test_llm_failure_falls_back_to_deterministic(self):
        """LLM call raising an exception → fallback with router_method='llm_fallback'."""
        judge = MagicMock()
        judge.with_structured_output.side_effect = RuntimeError("network error")
        judge.invoke.side_effect = RuntimeError("network error")

        decision = route_message_hybrid(
            "find the clause",
            has_attachments=False,
            judge_llm=judge,
            llm_confidence_threshold=0.80,
        )
        assert decision.router_method == "llm_fallback"
        assert decision.route in {"BASIC", "AGENT"}  # deterministic fallback


# ---------------------------------------------------------------------------
# Text parsing fallback
# ---------------------------------------------------------------------------

class TestParseResponseText:
    def test_parse_valid_json(self):
        text = '{"route": "AGENT", "confidence": 0.92, "reasoning": "doc query", "suggested_agent": "rag_agent"}'
        out = _parse_llm_response_text(text)
        assert out.route == "AGENT"
        assert out.confidence == 0.92
        assert out.suggested_agent == "rag_agent"

    def test_parse_missing_route_uses_keyword_fallback(self):
        out = _parse_llm_response_text("This is a document retrieval request")
        assert out.route == "AGENT"

    def test_parse_basic_keyword(self):
        out = _parse_llm_response_text("This looks like a simple greeting, route to basic")
        assert out.route == "BASIC"

    def test_confidence_clamped_to_range(self):
        text = '{"route": "AGENT", "confidence": 1.5, "reasoning": "test"}'
        out = _parse_llm_response_text(text)
        assert out.confidence <= 1.0

    def test_unknown_suggested_agent_cleared(self):
        text = '{"route": "AGENT", "confidence": 0.8, "reasoning": "test", "suggested_agent": "bogus_agent"}'
        out = _parse_llm_response_text(text)
        assert out.suggested_agent == ""
