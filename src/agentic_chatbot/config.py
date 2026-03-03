from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # --- Providers ---
    llm_provider: str  # ollama | azure
    embeddings_provider: str  # ollama | azure

    # --- Ollama ---
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    ollama_temperature: float
    ollama_num_predict: int

    # --- Azure OpenAI (optional) ---
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str | None
    azure_openai_chat_deployment: str | None
    azure_openai_embed_deployment: str | None
    azure_temperature: float

    # --- Runtime limits ---
    max_agent_steps: int
    max_tool_calls: int

    # --- RAG defaults ---
    rag_top_k_vector: int
    rag_top_k_keyword: int
    rag_max_retries: int
    rag_min_evidence_chunks: int
    max_rag_agent_steps: int  # step budget for the RAG loop agent

    # --- Text splitting ---
    chunk_size: int
    chunk_overlap: int

    # --- PostgreSQL ---
    pg_dsn: str          # e.g. "postgresql://user:pass@localhost:5432/ragdb"
    embedding_dim: int   # must match embedding model output (768 nomic, 1536 ada-002)

    # --- Storage paths (files only, no more vector/bm25 index dirs) ---
    project_root: Path
    data_dir: Path
    kb_dir: Path
    uploads_dir: Path

    # --- Multi-agent graph ---
    supervisor_max_loops: int        # env: SUPERVISOR_MAX_LOOPS (default: 5)
    max_parallel_rag_workers: int    # env: MAX_PARALLEL_RAG_WORKERS (default: 4)
    enable_parallel_rag: bool        # env: ENABLE_PARALLEL_RAG (default: True)

    # --- Scratchpad ---
    clear_scratchpad_per_turn: bool  # wipe session.scratchpad after each turn

    # --- OCR (PaddleOCR, optional) ---
    ocr_enabled: bool        # env: USE_PADDLE_OCR (default: True)
    ocr_language: str        # env: OCR_LANGUAGE   (default: "en")
    ocr_use_gpu: bool        # env: OCR_USE_GPU    (default: False)
    ocr_min_page_chars: int  # env: OCR_MIN_PAGE_CHARS (default: 50)
                             # PDF pages with fewer extracted chars trigger OCR fallback

    # --- Langfuse (optional) ---
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_debug: bool


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _as_int(name: str, default: int) -> int:
    v = _getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    v = _getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    v = _getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


def load_settings(dotenv_path: str | None = None) -> Settings:
    """Load settings from environment (and optional .env)."""

    load_dotenv(dotenv_path=dotenv_path)

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    llm_provider = str(_getenv("LLM_PROVIDER", "ollama")).lower()
    embeddings_provider = str(_getenv("EMBEDDINGS_PROVIDER", llm_provider)).lower()

    # Ollama
    ollama_base_url = str(_getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model = str(_getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b"))
    ollama_embed_model = str(_getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_temperature = _as_float("OLLAMA_TEMPERATURE", 0.2)
    ollama_num_predict = _as_int("OLLAMA_NUM_PREDICT", 512)

    # Azure OpenAI (optional)
    azure_openai_api_key = _getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = _getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version = _getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    azure_openai_chat_deployment = _getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embed_deployment = _getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    azure_temperature = _as_float("AZURE_TEMPERATURE", 0.2)

    # Runtime
    max_agent_steps = _as_int("MAX_AGENT_STEPS", 10)
    max_tool_calls = _as_int("MAX_TOOL_CALLS", 12)

    # RAG
    rag_top_k_vector = _as_int("RAG_TOPK_VECTOR", 12)
    rag_top_k_keyword = _as_int("RAG_TOPK_BM25", 12)
    rag_max_retries = _as_int("RAG_MAX_RETRIES", 2)
    rag_min_evidence_chunks = _as_int("RAG_MIN_EVIDENCE_CHUNKS", 2)
    max_rag_agent_steps = _as_int("MAX_RAG_AGENT_STEPS", 8)

    # Text splitting
    chunk_size = _as_int("CHUNK_SIZE", 900)
    chunk_overlap = _as_int("CHUNK_OVERLAP", 150)

    # PostgreSQL
    pg_dsn = str(_getenv("PG_DSN", "postgresql://localhost:5432/ragdb"))
    embedding_dim = _as_int("EMBEDDING_DIM", 768)

    # Paths
    kb_dir = Path(_getenv("KB_DIR", str(data_dir / "kb")))
    uploads_dir = Path(_getenv("UPLOADS_DIR", str(data_dir / "uploads")))

    # Multi-agent graph
    supervisor_max_loops = _as_int("SUPERVISOR_MAX_LOOPS", 5)
    max_parallel_rag_workers = _as_int("MAX_PARALLEL_RAG_WORKERS", 4)
    enable_parallel_rag = _as_bool("ENABLE_PARALLEL_RAG", True)

    # Scratchpad
    clear_scratchpad_per_turn = _as_bool("CLEAR_SCRATCHPAD_PER_TURN", True)

    # OCR
    ocr_enabled        = _as_bool("USE_PADDLE_OCR", True)
    ocr_language       = str(_getenv("OCR_LANGUAGE", "en"))
    ocr_use_gpu        = _as_bool("OCR_USE_GPU", False)
    ocr_min_page_chars = _as_int("OCR_MIN_PAGE_CHARS", 50)

    # Langfuse
    langfuse_host = _getenv("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_public_key = _getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = _getenv("LANGFUSE_SECRET_KEY")
    langfuse_debug = _as_bool("LANGFUSE_DEBUG", False)

    # Ensure base directories exist
    for p in [data_dir, kb_dir, uploads_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return Settings(
        llm_provider=llm_provider,
        embeddings_provider=embeddings_provider,
        ollama_base_url=ollama_base_url,
        ollama_chat_model=ollama_chat_model,
        ollama_embed_model=ollama_embed_model,
        ollama_temperature=ollama_temperature,
        ollama_num_predict=ollama_num_predict,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_api_version=azure_openai_api_version,
        azure_openai_chat_deployment=azure_openai_chat_deployment,
        azure_openai_embed_deployment=azure_openai_embed_deployment,
        azure_temperature=azure_temperature,
        max_agent_steps=max_agent_steps,
        max_tool_calls=max_tool_calls,
        rag_top_k_vector=rag_top_k_vector,
        rag_top_k_keyword=rag_top_k_keyword,
        rag_max_retries=rag_max_retries,
        rag_min_evidence_chunks=rag_min_evidence_chunks,
        max_rag_agent_steps=max_rag_agent_steps,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pg_dsn=pg_dsn,
        embedding_dim=embedding_dim,
        project_root=project_root,
        data_dir=data_dir,
        kb_dir=kb_dir,
        uploads_dir=uploads_dir,
        supervisor_max_loops=supervisor_max_loops,
        max_parallel_rag_workers=max_parallel_rag_workers,
        enable_parallel_rag=enable_parallel_rag,
        clear_scratchpad_per_turn=clear_scratchpad_per_turn,
        ocr_enabled=ocr_enabled,
        ocr_language=ocr_language,
        ocr_use_gpu=ocr_use_gpu,
        ocr_min_page_chars=ocr_min_page_chars,
        langfuse_host=langfuse_host,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_debug=langfuse_debug,
    )
