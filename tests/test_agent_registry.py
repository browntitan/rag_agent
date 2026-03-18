"""Unit tests for AgentRegistry and AgentSpec."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_chatbot.agents.agent_registry import AgentRegistry, AgentSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings():
    """Return a minimal mock Settings object."""
    s = MagicMock()
    return s


# ---------------------------------------------------------------------------
# AgentSpec
# ---------------------------------------------------------------------------

class TestAgentSpec:
    def test_frozen_dataclass(self):
        spec = AgentSpec(
            name="test_agent",
            display_name="Test Agent",
            description="Does test things.",
            use_when=["Testing"],
            skills_key="test_agent",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # frozen — must raise

    def test_enabled_default_true(self):
        spec = AgentSpec(
            name="x",
            display_name="X",
            description="Desc",
            use_when=["anything"],
            skills_key="x",
        )
        assert spec.enabled is True

    def test_enabled_can_be_false(self):
        spec = AgentSpec(
            name="x",
            display_name="X",
            description="Desc",
            use_when=[],
            skills_key="x",
            enabled=False,
        )
        assert spec.enabled is False


# ---------------------------------------------------------------------------
# AgentRegistry — built-in agents
# ---------------------------------------------------------------------------

class TestBuiltinAgentsAlwaysRegistered:
    """rag_agent, utility_agent, and parallel_rag must always be registered."""

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_rag_agent_present(self, _mock):
        registry = AgentRegistry(_make_settings())
        assert registry.get("rag_agent") is not None

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_utility_agent_present(self, _mock):
        registry = AgentRegistry(_make_settings())
        assert registry.get("utility_agent") is not None

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_parallel_rag_present(self, _mock):
        registry = AgentRegistry(_make_settings())
        assert registry.get("parallel_rag") is not None

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_three_core_agents_enabled(self, _mock):
        registry = AgentRegistry(_make_settings())
        names = {a.name for a in registry.list_enabled()}
        assert {"rag_agent", "utility_agent", "parallel_rag"}.issubset(names)


# ---------------------------------------------------------------------------
# AgentRegistry — data_analyst Docker availability
# ---------------------------------------------------------------------------

class TestDataAnalystDockerAvailability:
    @patch.object(AgentRegistry, "_check_docker_available", return_value=True)
    def test_data_analyst_enabled_when_docker_available(self, _mock):
        registry = AgentRegistry(_make_settings())
        spec = registry.get("data_analyst")
        assert spec is not None
        assert spec.enabled is True

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_data_analyst_disabled_when_docker_unavailable(self, _mock):
        registry = AgentRegistry(_make_settings())
        spec = registry.get("data_analyst")
        assert spec is not None
        assert spec.enabled is False

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_data_analyst_not_in_list_enabled(self, _mock):
        registry = AgentRegistry(_make_settings())
        names = {a.name for a in registry.list_enabled()}
        assert "data_analyst" not in names

    @patch.object(AgentRegistry, "_check_docker_available", return_value=True)
    def test_data_analyst_in_list_enabled_when_docker_ok(self, _mock):
        registry = AgentRegistry(_make_settings())
        names = {a.name for a in registry.list_enabled()}
        assert "data_analyst" in names


# ---------------------------------------------------------------------------
# AgentRegistry — valid_agent_names
# ---------------------------------------------------------------------------

class TestValidAgentNames:
    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_always_includes_end(self, _mock):
        registry = AgentRegistry(_make_settings())
        assert "__end__" in registry.valid_agent_names()

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_includes_enabled_agents(self, _mock):
        registry = AgentRegistry(_make_settings())
        valid = registry.valid_agent_names()
        assert "rag_agent" in valid
        assert "utility_agent" in valid
        assert "parallel_rag" in valid

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_excludes_disabled_agents(self, _mock):
        registry = AgentRegistry(_make_settings())
        # data_analyst is disabled when docker unavailable
        assert "data_analyst" not in registry.valid_agent_names()

    @patch.object(AgentRegistry, "_check_docker_available", return_value=True)
    def test_includes_data_analyst_when_docker_ok(self, _mock):
        registry = AgentRegistry(_make_settings())
        assert "data_analyst" in registry.valid_agent_names()


# ---------------------------------------------------------------------------
# AgentRegistry — format_for_supervisor_prompt
# ---------------------------------------------------------------------------

class TestFormatForSupervisorPrompt:
    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_contains_enabled_agent_names(self, _mock):
        registry = AgentRegistry(_make_settings())
        prompt = registry.format_for_supervisor_prompt()
        assert "rag_agent" in prompt
        assert "utility_agent" in prompt
        assert "parallel_rag" in prompt

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_does_not_contain_disabled_agent(self, _mock):
        registry = AgentRegistry(_make_settings())
        prompt = registry.format_for_supervisor_prompt()
        # data_analyst is disabled → should not appear in prompt
        assert "data_analyst" not in prompt

    @patch.object(AgentRegistry, "_check_docker_available", return_value=True)
    def test_contains_data_analyst_when_docker_ok(self, _mock):
        registry = AgentRegistry(_make_settings())
        prompt = registry.format_for_supervisor_prompt()
        assert "data_analyst" in prompt

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_contains_use_when_bullets(self, _mock):
        registry = AgentRegistry(_make_settings())
        prompt = registry.format_for_supervisor_prompt()
        # Each agent has "Use when:" section
        assert "Use when:" in prompt

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_has_available_agents_header(self, _mock):
        registry = AgentRegistry(_make_settings())
        prompt = registry.format_for_supervisor_prompt()
        assert "## Available Agents" in prompt


# ---------------------------------------------------------------------------
# AgentRegistry — register custom agent
# ---------------------------------------------------------------------------

class TestRegisterCustomAgent:
    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_register_new_agent(self, _mock):
        registry = AgentRegistry(_make_settings())
        custom = AgentSpec(
            name="custom_agent",
            display_name="Custom Agent",
            description="Does custom things.",
            use_when=["Custom use case"],
            skills_key="custom_agent",
        )
        registry.register(custom)
        assert registry.get("custom_agent") is not None

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_custom_agent_appears_in_list_enabled(self, _mock):
        registry = AgentRegistry(_make_settings())
        custom = AgentSpec(
            name="custom_agent",
            display_name="Custom Agent",
            description="Does custom things.",
            use_when=["Custom use case"],
            skills_key="custom_agent",
            enabled=True,
        )
        registry.register(custom)
        names = {a.name for a in registry.list_enabled()}
        assert "custom_agent" in names

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_disabled_custom_agent_not_in_list_enabled(self, _mock):
        registry = AgentRegistry(_make_settings())
        custom = AgentSpec(
            name="custom_agent",
            display_name="Custom Agent",
            description="Does custom things.",
            use_when=[],
            skills_key="custom_agent",
            enabled=False,
        )
        registry.register(custom)
        names = {a.name for a in registry.list_enabled()}
        assert "custom_agent" not in names

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_re_register_replaces_spec(self, _mock):
        registry = AgentRegistry(_make_settings())
        v1 = AgentSpec(
            name="my_agent", display_name="V1", description="V1 desc",
            use_when=[], skills_key="my_agent",
        )
        v2 = AgentSpec(
            name="my_agent", display_name="V2", description="V2 desc",
            use_when=[], skills_key="my_agent",
        )
        registry.register(v1)
        registry.register(v2)
        assert registry.get("my_agent").display_name == "V2"

    @patch.object(AgentRegistry, "_check_docker_available", return_value=False)
    def test_custom_agent_in_valid_names(self, _mock):
        registry = AgentRegistry(_make_settings())
        registry.register(AgentSpec(
            name="x_agent", display_name="X", description="X",
            use_when=[], skills_key="x_agent", enabled=True,
        ))
        assert "x_agent" in registry.valid_agent_names()


# ---------------------------------------------------------------------------
# AgentRegistry — _check_docker_available
# ---------------------------------------------------------------------------

class TestCheckDockerAvailable:
    def test_returns_false_when_docker_not_installed(self):
        with patch.dict("sys.modules", {"docker": None}):
            registry = AgentRegistry.__new__(AgentRegistry)
            registry._settings = _make_settings()
            result = registry._check_docker_available()
            assert result is False

    def test_returns_false_on_exception(self):
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = Exception("Docker not running")
        with patch.dict("sys.modules", {"docker": mock_docker}):
            registry = AgentRegistry.__new__(AgentRegistry)
            registry._settings = _make_settings()
            result = registry._check_docker_available()
            assert result is False
