from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class NotebookSettings:
    provider_mode: str  # azure | ollama | vllm | nvidia

    # Shared runtime
    repo_root: Path
    demo_root: Path
    kb_dir: Path
    skills_dir: Path
    pg_dsn: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    rag_top_k_vector: int
    rag_top_k_keyword: int
    max_agent_steps: int
    max_tool_calls: int
    temperature: float
    judge_temperature: float
    http2_enabled: bool
    ssl_verify: bool
    ssl_cert_file: Optional[Path]
    tiktoken_enabled: bool
    tiktoken_cache_dir: Optional[Path]

    # Skills / showcase toggles
    skills_enabled: bool
    skills_showcase_mode: bool

    # Azure
    azure_endpoint: Optional[str]
    azure_api_key: Optional[str]
    azure_api_version: str
    azure_chat_deployment: Optional[str]
    azure_judge_deployment: Optional[str]
    azure_embed_deployment: Optional[str]

    # Ollama
    ollama_base_url: str
    ollama_chat_model: str
    ollama_judge_model: str
    ollama_embed_model: str
    ollama_num_predict: int

    # vLLM (OpenAI-compatible)
    vllm_base_url: Optional[str]
    vllm_api_key: Optional[str]
    vllm_chat_model: Optional[str]
    vllm_judge_model: Optional[str]
    vllm_embed_model: Optional[str]
    vllm_use_openai_embeddings: bool

    # NVIDIA OpenAI-compatible endpoint (chat/judge only)
    nvidia_endpoint: Optional[str]
    nvidia_token: Optional[str]
    nvidia_chat_model: Optional[str]
    nvidia_judge_model: Optional[str]
    nvidia_temperature: float
    nvidia_embeddings_backend: str  # ollama | azure | localhash


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _as_int(name: str, default: int) -> int:
    raw = _getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    raw = _getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    raw = _getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_settings(dotenv_path: Optional[str] = None) -> NotebookSettings:
    load_dotenv(dotenv_path=dotenv_path)

    # runtime/config.py -> runtime -> demo_notebook -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    demo_root = Path(__file__).resolve().parents[1]

    provider_mode = str(_getenv("NOTEBOOK_PROVIDER", "azure")).lower()
    if provider_mode not in {"azure", "ollama", "vllm", "nvidia"}:
        raise ValueError("NOTEBOOK_PROVIDER must be one of: azure, ollama, vllm, nvidia")

    nvidia_embeddings_backend = str(_getenv("NOTEBOOK_NVIDIA_EMBEDDINGS_BACKEND", "ollama")).lower()
    if nvidia_embeddings_backend not in {"ollama", "azure", "localhash"}:
        raise ValueError("NOTEBOOK_NVIDIA_EMBEDDINGS_BACKEND must be one of: ollama, azure, localhash")

    kb_dir = _resolve_path(
        str(_getenv("NOTEBOOK_KB_DIR", "../data/kb")),
        base=demo_root,
    )
    skills_dir = _resolve_path(
        str(_getenv("NOTEBOOK_SKILLS_DIR", "./skills")),
        base=demo_root,
    )
    ssl_cert_raw = _getenv("NOTEBOOK_SSL_CERT_FILE")
    ssl_cert_file = _resolve_path(ssl_cert_raw, base=demo_root) if ssl_cert_raw else None
    tiktoken_cache_raw = _getenv("NOTEBOOK_TIKTOKEN_CACHE_DIR")
    tiktoken_cache_dir = _resolve_path(tiktoken_cache_raw, base=demo_root) if tiktoken_cache_raw else None

    if ssl_cert_file and _as_bool("NOTEBOOK_SSL_VERIFY", True):
        # Ensure non-httpx code paths (e.g. tiktoken/urllib) use the same CA bundle.
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        os.environ["REQUESTS_CA_BUNDLE"] = str(ssl_cert_file)
        os.environ["CURL_CA_BUNDLE"] = str(ssl_cert_file)

    if tiktoken_cache_dir:
        tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

    settings = NotebookSettings(
        provider_mode=provider_mode,
        repo_root=repo_root,
        demo_root=demo_root,
        kb_dir=kb_dir,
        skills_dir=skills_dir,
        pg_dsn=str(_getenv("NOTEBOOK_PG_DSN", "postgresql://raguser:ragpass@localhost:5432/ragdb")),
        embedding_dim=_as_int("NOTEBOOK_EMBEDDING_DIM", 1536),
        chunk_size=_as_int("NOTEBOOK_CHUNK_SIZE", 900),
        chunk_overlap=_as_int("NOTEBOOK_CHUNK_OVERLAP", 120),
        rag_top_k_vector=_as_int("NOTEBOOK_RAG_TOP_K_VECTOR", 8),
        rag_top_k_keyword=_as_int("NOTEBOOK_RAG_TOP_K_KEYWORD", 8),
        max_agent_steps=_as_int("NOTEBOOK_MAX_AGENT_STEPS", 8),
        max_tool_calls=_as_int("NOTEBOOK_MAX_TOOL_CALLS", 10),
        temperature=_as_float("NOTEBOOK_TEMPERATURE", 0.2),
        judge_temperature=_as_float("NOTEBOOK_JUDGE_TEMPERATURE", 0.0),
        http2_enabled=_as_bool("NOTEBOOK_HTTP2", True),
        ssl_verify=_as_bool("NOTEBOOK_SSL_VERIFY", True),
        ssl_cert_file=ssl_cert_file,
        tiktoken_enabled=_as_bool("NOTEBOOK_TIKTOKEN_ENABLED", True),
        tiktoken_cache_dir=tiktoken_cache_dir,
        skills_enabled=_as_bool("NOTEBOOK_SKILLS_ENABLED", True),
        skills_showcase_mode=_as_bool("NOTEBOOK_SKILLS_SHOWCASE_MODE", False),
        azure_endpoint=_getenv("NOTEBOOK_AZURE_ENDPOINT"),
        azure_api_key=_getenv("NOTEBOOK_AZURE_API_KEY"),
        azure_api_version=str(_getenv("NOTEBOOK_AZURE_API_VERSION", "2024-05-01-preview")),
        azure_chat_deployment=_getenv("NOTEBOOK_AZURE_CHAT_DEPLOYMENT"),
        azure_judge_deployment=_getenv("NOTEBOOK_AZURE_JUDGE_DEPLOYMENT"),
        azure_embed_deployment=_getenv("NOTEBOOK_AZURE_EMBED_DEPLOYMENT"),
        ollama_base_url=str(_getenv("NOTEBOOK_OLLAMA_BASE_URL", "http://localhost:11434")),
        ollama_chat_model=str(_getenv("NOTEBOOK_OLLAMA_CHAT_MODEL", "qwen3:8b")),
        ollama_judge_model=str(_getenv("NOTEBOOK_OLLAMA_JUDGE_MODEL", "qwen3:8b")),
        ollama_embed_model=str(_getenv("NOTEBOOK_OLLAMA_EMBED_MODEL", "nomic-embed-text")),
        ollama_num_predict=_as_int("NOTEBOOK_OLLAMA_NUM_PREDICT", 1024),
        vllm_base_url=_getenv("NOTEBOOK_VLLM_BASE_URL"),
        vllm_api_key=_getenv("NOTEBOOK_VLLM_API_KEY", "not-required"),
        vllm_chat_model=_getenv("NOTEBOOK_VLLM_CHAT_MODEL"),
        vllm_judge_model=_getenv("NOTEBOOK_VLLM_JUDGE_MODEL"),
        vllm_embed_model=_getenv("NOTEBOOK_VLLM_EMBED_MODEL"),
        vllm_use_openai_embeddings=_as_bool("NOTEBOOK_VLLM_USE_OPENAI_EMBEDDINGS", False),
        nvidia_endpoint=_getenv("NOTEBOOK_NVIDIA_ENDPOINT"),
        nvidia_token=_getenv("NOTEBOOK_NVIDIA_TOKEN", _getenv("Token")),
        nvidia_chat_model=_getenv("NOTEBOOK_NVIDIA_CHAT_MODEL"),
        nvidia_judge_model=_getenv("NOTEBOOK_NVIDIA_JUDGE_MODEL", _getenv("NOTEBOOK_NVIDIA_CHAT_MODEL")),
        nvidia_temperature=_as_float("NOTEBOOK_NVIDIA_TEMPERATURE", 0.0),
        nvidia_embeddings_backend=nvidia_embeddings_backend,
    )

    return settings
