from __future__ import annotations

from agentic_chatbot.config import load_settings


def test_deprecated_runtime_compat_env_vars_no_longer_block_settings_load(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "unexpected_legacy_value")
    monkeypatch.setenv("AGENT_DEFINITIONS_JSON", "{\"general\": {\"name\": \"ignored\"}}")

    settings = load_settings()

    assert settings.agent_runtime_mode == "unexpected_legacy_value"
    assert settings.agent_definitions_json == "{\"general\": {\"name\": \"ignored\"}}"
