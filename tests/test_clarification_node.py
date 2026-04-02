"""Tests for the clarification node and related supervisor routing changes."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Clarification node unit tests
# ---------------------------------------------------------------------------

class TestClarificationNode:
    def _make_state(self, question: str = "", needs: bool = True) -> dict:
        return {
            "messages": [HumanMessage(content="summarise")],
            "needs_clarification": needs,
            "clarification_question": question,
            "next_agent": "clarify",
            "final_answer": "",
            "rag_sub_tasks": [],
            "rag_results": [],
        }

    def test_emits_ai_message_with_question(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        question = "Which document would you like me to summarise?"
        result = node(self._make_state(question=question))

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert question in msg.content

    def test_uses_default_question_when_empty(self):
        from agentic_chatbot.graph.nodes.clarification_node import (
            make_clarification_node,
            _DEFAULT_QUESTION,
        )

        node = make_clarification_node()
        result = node(self._make_state(question=""))

        msg = result["messages"][0]
        assert _DEFAULT_QUESTION in msg.content

    def test_resets_needs_clarification_flag(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        result = node(self._make_state(question="What document?"))

        assert result["needs_clarification"] is False

    def test_resets_clarification_question(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        result = node(self._make_state(question="What document?"))

        assert result["clarification_question"] == ""

    def test_routes_to_end(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        result = node(self._make_state(question="Please clarify."))

        assert result["next_agent"] == "__end__"

    def test_final_answer_equals_question(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        question = "Which document are you referring to?"
        result = node(self._make_state(question=question))

        assert result["final_answer"] == question

    def test_strips_whitespace_from_question(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        result = node(self._make_state(question="   Please clarify.   "))

        assert result["messages"][0].content == "Please clarify."

    def test_make_returns_callable(self):
        from agentic_chatbot.graph.nodes.clarification_node import make_clarification_node

        node = make_clarification_node()
        assert callable(node)

    def test_handles_none_question_gracefully(self):
        from agentic_chatbot.graph.nodes.clarification_node import (
            make_clarification_node,
            _DEFAULT_QUESTION,
        )

        node = make_clarification_node()
        state = self._make_state()
        state["clarification_question"] = None  # type: ignore[assignment]
        result = node(state)

        assert _DEFAULT_QUESTION in result["messages"][0].content


# ---------------------------------------------------------------------------
# Supervisor response parsing — clarify route
# ---------------------------------------------------------------------------

class TestSupervisorClarifyParsing:
    def test_parse_clarify_agent(self):
        import json
        from agentic_chatbot.graph.supervisor import _parse_supervisor_response

        payload = {
            "reasoning": "No document context",
            "next_agent": "clarify",
            "clarification_question": "Which document are you referring to?",
            "direct_answer": "",
        }
        result = _parse_supervisor_response(json.dumps(payload))

        assert result["next_agent"] == "clarify"
        assert result["clarification_question"] == "Which document are you referring to?"

    def test_clarify_in_valid_agents(self):
        from agentic_chatbot.graph.supervisor import _VALID_AGENTS

        assert "clarify" in _VALID_AGENTS

    def test_clarify_not_remapped_to_rag(self):
        """Supervisor should NOT fall back to rag_agent when next_agent is 'clarify'."""
        import json
        from agentic_chatbot.graph.supervisor import _parse_supervisor_response

        payload = {
            "reasoning": "Need more info",
            "next_agent": "clarify",
            "clarification_question": "What file?",
        }
        result = _parse_supervisor_response(json.dumps(payload), valid_agents={"clarify", "__end__", "rag_agent"})
        assert result["next_agent"] == "clarify"


# ---------------------------------------------------------------------------
# State schema includes new fields
# ---------------------------------------------------------------------------

class TestAgentStateFields:
    def test_needs_clarification_default(self):
        from agentic_chatbot.graph.state import AgentState

        state = AgentState()
        assert state.get("needs_clarification", False) is False

    def test_clarification_question_default(self):
        from agentic_chatbot.graph.state import AgentState

        state = AgentState()
        assert state.get("clarification_question", "") == ""
