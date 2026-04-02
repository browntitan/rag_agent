"""Tests for the evaluator node (Generator-Evaluator pattern)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(final_answer: str = "The contract states X.", next_agent: str = "rag_agent",
                eval_retry_count: int = 0, messages=None) -> dict:
    if messages is None:
        messages = [HumanMessage(content="What does the contract say?")]
    return {
        "messages": messages,
        "final_answer": final_answer,
        "next_agent": next_agent,
        "eval_retry_count": eval_retry_count,
        "evaluation_result": "",
        "rag_results": [],
        "rag_sub_tasks": [],
    }


def _mock_llm_pass() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='{"pass": true, "failures": [], "suggestion": ""}'
    )
    return llm


def _mock_llm_fail(suggestion: str = "Try hybrid search") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=f'{{"pass": false, "failures": ["relevance: answer is off-topic"], "suggestion": "{suggestion}"}}'
    )
    return llm


class TestEvaluatorNode:
    def test_pass_returns_evaluation_result_pass(self):
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_pass())
        result = node(_make_state())

        assert result.get("evaluation_result") == "pass"

    def test_fail_clears_final_answer(self):
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state())

        assert result.get("evaluation_result") == "fail"
        assert result.get("final_answer") == ""

    def test_fail_increments_retry_count(self):
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state(eval_retry_count=0))

        assert result.get("eval_retry_count") == 1

    def test_fail_adds_ai_message(self):
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail("Use keyword search"))
        result = node(_make_state())

        messages = result.get("messages", [])
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert "Use keyword search" in messages[0].content

    def test_max_retry_cap_accepts_answer(self):
        """After eval_retry_count >= 1, accept answer regardless of LLM response."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state(eval_retry_count=1))

        assert result.get("evaluation_result") == "pass"

    def test_skips_non_rag_agents(self):
        """Utility and data_analyst outputs bypass evaluation (return empty dict)."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state(next_agent="utility_agent"))

        assert result == {}

    def test_skips_short_answers(self):
        """Very short or empty final_answer skips evaluation."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state(final_answer=""))

        assert result == {}

    def test_llm_exception_passes_through(self):
        """If evaluation LLM call fails, the answer is accepted (don't block user)."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        node = make_evaluator_node(llm)
        result = node(_make_state())

        assert result.get("evaluation_result") == "pass"

    def test_no_question_in_messages_passes_through(self):
        """If no HumanMessage found, skip evaluation and pass."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        node = make_evaluator_node(_mock_llm_fail())
        result = node(_make_state(messages=[AIMessage(content="previous response")]))

        assert result.get("evaluation_result") == "pass"

    def test_custom_criteria(self):
        """Custom criteria dict is accepted and used."""
        from agentic_chatbot.graph.nodes.evaluator_node import make_evaluator_node

        custom = {"accuracy": "Is the answer accurate?"}
        node = make_evaluator_node(_mock_llm_pass(), criteria=custom)
        result = node(_make_state())

        assert result.get("evaluation_result") == "pass"


class TestParseEvalResponse:
    def test_clean_json(self):
        from agentic_chatbot.graph.nodes.evaluator_node import _parse_eval_response

        result = _parse_eval_response('{"pass": true, "failures": [], "suggestion": ""}')
        assert result["pass"] is True

    def test_json_in_markdown_block(self):
        from agentic_chatbot.graph.nodes.evaluator_node import _parse_eval_response

        content = '```json\n{"pass": false, "failures": ["x"], "suggestion": "try harder"}\n```'
        result = _parse_eval_response(content)
        assert result["pass"] is False

    def test_fallback_on_fail_keyword(self):
        from agentic_chatbot.graph.nodes.evaluator_node import _parse_eval_response

        result = _parse_eval_response("FAIL — the answer was completely off-topic.")
        assert result["pass"] is False

    def test_fallback_on_unparseable_defaults_pass(self):
        from agentic_chatbot.graph.nodes.evaluator_node import _parse_eval_response

        result = _parse_eval_response("This is some garbled response with no JSON")
        assert result["pass"] is True
