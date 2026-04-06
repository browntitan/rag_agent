from __future__ import annotations

from agentic_chatbot_next.config import load_settings


def test_deprecated_runtime_compat_env_vars_no_longer_block_settings_load(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "unexpected_legacy_value")
    monkeypatch.setenv("AGENT_DEFINITIONS_JSON", "{\"general\": {\"name\": \"ignored\"}}")

    settings = load_settings()

    assert settings.agent_runtime_mode == "unexpected_legacy_value"
    assert settings.agent_definitions_json == "{\"general\": {\"name\": \"ignored\"}}"


def test_agent_model_override_envs_are_parsed_and_normalized(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_GENERAL_CHAT_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("AGENT_DATA_ANALYST_CHAT_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("AGENT_MEMORY_MAINTAINER_JUDGE_MODEL", "gpt-oss:20b")

    settings = load_settings()

    assert settings.agent_chat_model_overrides["general"] == "gpt-oss:20b"
    assert settings.agent_chat_model_overrides["data_analyst"] == "gpt-oss:20b"
    assert settings.agent_judge_model_overrides["memory_maintainer"] == "gpt-oss:20b"
