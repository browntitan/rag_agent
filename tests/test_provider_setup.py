from __future__ import annotations

from pathlib import Path

from agentic_chatbot.config import load_settings
from agentic_chatbot.providers import validate_provider_configuration


_ENV_KEYS = [
    "DATA_DIR",
    "LLM_PROVIDER",
    "JUDGE_PROVIDER",
    "EMBEDDINGS_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_JUDGE_MODEL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_TEMPERATURE",
    "OLLAMA_NUM_PREDICT",
    "DEMO_OLLAMA_NUM_PREDICT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_JUDGE_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT",
    "EMBEDDING_DIM",
    "SSL_CERT_FILE",
    "NVIDIA_OPENAI_ENDPOINT",
    "NVIDIA_API_TOKEN",
    "NVIDIA_CHAT_MODEL",
    "NVIDIA_JUDGE_MODEL",
]


def _load_test_settings(tmp_path: Path, monkeypatch, lines: list[str]):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    env_path = tmp_path / ".env.test"
    env_path.write_text("\n".join([f"DATA_DIR={tmp_path / 'data'}", *lines]) + "\n", encoding="utf-8")
    return load_settings(dotenv_path=str(env_path))


def test_validate_provider_configuration_accepts_ollama_only_with_blank_azure(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=ollama",
            "JUDGE_PROVIDER=ollama",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=",
            "AZURE_OPENAI_ENDPOINT=",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_validate_provider_configuration_accepts_azure_only_with_blank_ollama(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=azure",
            "OLLAMA_BASE_URL=",
            "OLLAMA_CHAT_MODEL=",
            "OLLAMA_JUDGE_MODEL=",
            "OLLAMA_EMBED_MODEL=",
            "AZURE_OPENAI_API_KEY=test-key",
            "AZURE_OPENAI_ENDPOINT=https://example-resource.openai.azure.com/",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002",
            "EMBEDDING_DIM=1536",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_validate_provider_configuration_accepts_mixed_azure_chat_and_ollama_embeddings(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=test-key",
            "AZURE_OPENAI_ENDPOINT=https://example-resource.openai.azure.com/",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_embeddings_only_validation_ignores_missing_chat_and_judge_azure_settings(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=",
            "AZURE_OPENAI_ENDPOINT=",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    scoped_issues = validate_provider_configuration(settings, contexts=("embeddings",))
    full_issues = validate_provider_configuration(settings)

    assert scoped_issues == []
    assert any(issue.context == "azure" for issue in full_issues)
    assert any(issue.context == "llm" for issue in full_issues)
    assert any(issue.context == "judge" for issue in full_issues)
