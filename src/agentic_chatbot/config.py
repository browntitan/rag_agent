from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # --- Backend selection ---
    database_backend: str       # postgres
    vector_store_backend: str   # pgvector
    object_store_backend: str   # local | s3 | azure_blob (future)
    skills_backend: str         # local | s3 | azure_blob (future)
    prompts_backend: str        # local | s3 | azure_blob (future)

    # --- Providers ---
    llm_provider: str  # ollama | azure | nvidia
    embeddings_provider: str  # ollama | azure
    judge_provider: str  # ollama | azure | nvidia (defaults to llm_provider)

    # --- Ollama ---
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    ollama_judge_model: str
    ollama_temperature: float
    ollama_num_predict: int
    demo_ollama_num_predict: int

    # --- Azure OpenAI (optional) ---
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str | None
    azure_openai_chat_deployment: str | None
    azure_openai_judge_deployment: str | None
    azure_openai_embed_deployment: str | None
    azure_temperature: float
    judge_temperature: float
    nvidia_openai_endpoint: str | None
    nvidia_api_token: str | None
    nvidia_chat_model: str | None
    nvidia_judge_model: str | None
    nvidia_temperature: float
    http2_enabled: bool
    ssl_verify: bool
    ssl_cert_file: Path | None
    tiktoken_enabled: bool
    tiktoken_cache_dir: Path | None

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
    kb_source_uri: str
    uploads_source_uri: str

    # --- Prompt/skills locations ---
    skills_dir: Path
    prompts_dir: Path
    shared_skills_path: Path
    general_agent_skills_path: Path
    rag_agent_skills_path: Path
    supervisor_agent_skills_path: Path
    utility_agent_skills_path: Path
    basic_chat_skills_path: Path
    judge_grading_prompt_path: Path
    judge_rewrite_prompt_path: Path
    grounded_answer_prompt_path: Path
    rag_synthesis_prompt_path: Path
    parallel_rag_synthesis_prompt_path: Path

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

    # --- Context defaults (CLI/demo compatibility) ---
    default_tenant_id: str
    default_user_id: str
    default_conversation_id: str

    # --- OpenAI-compatible gateway ---
    gateway_model_id: str

    # --- LLM Router ---
    llm_router_enabled: bool            # env: LLM_ROUTER_ENABLED (default: True)
    llm_router_confidence_threshold: float  # env: LLM_ROUTER_CONFIDENCE_THRESHOLD (default: 0.70)

    # --- Web search fallback (opt-in) ---
    tavily_api_key: str | None          # env: TAVILY_API_KEY
    web_search_enabled: bool            # env: WEB_SEARCH_ENABLED (default: False)

    # --- Data Analyst / Sandbox ---
    sandbox_docker_image: str           # env: SANDBOX_DOCKER_IMAGE (default: "python:3.12-slim")
    sandbox_timeout_seconds: int        # env: SANDBOX_TIMEOUT_SECONDS (default: 60)
    sandbox_memory_limit: str           # env: SANDBOX_MEMORY_LIMIT (default: "512m")
    data_analyst_max_steps: int         # env: DATA_ANALYST_MAX_STEPS (default: 10)
    data_analyst_skills_path: Path      # constructed: skills_dir / "data_analyst_agent.md"

    # --- Session Workspace ---
    workspace_dir: Path                 # env: WORKSPACE_DIR (default: data/workspaces)
    workspace_session_ttl_hours: int    # env: WORKSPACE_SESSION_TTL_HOURS (default: 24; 0=keep forever)


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


def _resolve_path(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_settings(dotenv_path: str | None = None) -> Settings:
    """Load settings from environment (and optional .env)."""

    load_dotenv(dotenv_path=dotenv_path)

    # config.py -> agentic_chatbot -> src -> repo_root
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(_getenv("DATA_DIR", str(project_root / "data")))

    # Backends
    database_backend = str(_getenv("DATABASE_BACKEND", "postgres")).lower()
    vector_store_backend = str(_getenv("VECTOR_STORE_BACKEND", "pgvector")).lower()
    object_store_backend = str(_getenv("OBJECT_STORE_BACKEND", "local")).lower()
    skills_backend = str(_getenv("SKILLS_BACKEND", "local")).lower()
    prompts_backend = str(_getenv("PROMPTS_BACKEND", "local")).lower()

    llm_provider = str(_getenv("LLM_PROVIDER", "azure")).lower()
    embeddings_provider = str(_getenv("EMBEDDINGS_PROVIDER", llm_provider)).lower()
    judge_provider = str(_getenv("JUDGE_PROVIDER", llm_provider)).lower()

    # Ollama
    ollama_base_url = str(_getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model = str(_getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"))
    ollama_embed_model = str(_getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_judge_model = str(_getenv("OLLAMA_JUDGE_MODEL", ollama_chat_model))
    ollama_temperature = _as_float("OLLAMA_TEMPERATURE", 0.2)
    ollama_num_predict = _as_int("OLLAMA_NUM_PREDICT", 2048)
    demo_ollama_num_predict = _as_int("DEMO_OLLAMA_NUM_PREDICT", max(ollama_num_predict, 2048))

    # Azure OpenAI (optional)
    azure_openai_api_key = _getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = _getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version = _getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    azure_openai_chat_deployment = _getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", _getenv("AZURE_OPENAI_DEPLOYMENT"))
    azure_openai_judge_deployment = _getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", azure_openai_chat_deployment)
    azure_openai_embed_deployment = _getenv(
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        _getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    )
    azure_temperature = _as_float("AZURE_TEMPERATURE", 0.2)
    nvidia_openai_endpoint = _getenv("NVIDIA_OPENAI_ENDPOINT")
    nvidia_api_token = _getenv("NVIDIA_API_TOKEN", _getenv("Token"))
    nvidia_chat_model = _getenv("NVIDIA_CHAT_MODEL")
    nvidia_judge_model = _getenv("NVIDIA_JUDGE_MODEL", nvidia_chat_model)
    nvidia_temperature = _as_float("NVIDIA_TEMPERATURE", 0.0)
    judge_temperature = _as_float("JUDGE_TEMPERATURE", 0.0)
    http2_enabled = _as_bool("HTTP2_ENABLED", True)
    ssl_verify = _as_bool("SSL_VERIFY", True)
    ssl_cert_raw = _getenv("SSL_CERT_FILE", _getenv("APP_SSL_CERT_FILE"))
    ssl_cert_file = _resolve_path(ssl_cert_raw, base=project_root) if ssl_cert_raw else None
    tiktoken_enabled = _as_bool("TIKTOKEN_ENABLED", True)
    tiktoken_cache_raw = _getenv("TIKTOKEN_CACHE_DIR")
    tiktoken_cache_dir = _resolve_path(tiktoken_cache_raw, base=project_root) if tiktoken_cache_raw else None

    if ssl_cert_file and ssl_verify:
        # Ensure non-httpx paths (e.g. tiktoken/urllib) trust corporate CA bundle.
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        os.environ["REQUESTS_CA_BUNDLE"] = str(ssl_cert_file)
        os.environ["CURL_CA_BUNDLE"] = str(ssl_cert_file)

    if tiktoken_cache_dir:
        tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

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
    embedding_dim = _as_int("EMBEDDING_DIM", 1536)

    # Paths
    kb_dir = Path(_getenv("KB_DIR", str(data_dir / "kb")))
    uploads_dir = Path(_getenv("UPLOADS_DIR", str(data_dir / "uploads")))
    kb_source_uri = str(_getenv("KB_SOURCE_URI", f"file://{kb_dir}"))
    uploads_source_uri = str(_getenv("UPLOADS_SOURCE_URI", f"file://{uploads_dir}"))

    skills_dir = Path(_getenv("SKILLS_DIR", str(data_dir / "skills")))
    prompts_dir = Path(_getenv("PROMPTS_DIR", str(data_dir / "prompts")))

    shared_skills_path = Path(_getenv("SHARED_SKILLS_PATH", str(skills_dir / "skills.md")))
    general_agent_skills_path = Path(_getenv("GENERAL_AGENT_SKILLS_PATH", str(skills_dir / "general_agent.md")))
    rag_agent_skills_path = Path(_getenv("RAG_AGENT_SKILLS_PATH", str(skills_dir / "rag_agent.md")))
    supervisor_agent_skills_path = Path(_getenv("SUPERVISOR_AGENT_SKILLS_PATH", str(skills_dir / "supervisor_agent.md")))
    utility_agent_skills_path = Path(_getenv("UTILITY_AGENT_SKILLS_PATH", str(skills_dir / "utility_agent.md")))
    basic_chat_skills_path = Path(_getenv("BASIC_CHAT_SKILLS_PATH", str(skills_dir / "basic_chat.md")))

    judge_grading_prompt_path = Path(_getenv("JUDGE_GRADING_PROMPT_PATH", str(prompts_dir / "judge_grading.txt")))
    judge_rewrite_prompt_path = Path(_getenv("JUDGE_REWRITE_PROMPT_PATH", str(prompts_dir / "judge_rewrite.txt")))
    grounded_answer_prompt_path = Path(_getenv("GROUNDED_ANSWER_PROMPT_PATH", str(prompts_dir / "grounded_answer.txt")))
    rag_synthesis_prompt_path = Path(_getenv("RAG_SYNTHESIS_PROMPT_PATH", str(prompts_dir / "rag_synthesis.txt")))
    parallel_rag_synthesis_prompt_path = Path(_getenv("PARALLEL_RAG_SYNTHESIS_PROMPT_PATH", str(prompts_dir / "parallel_rag_synthesis.txt")))

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

    # Context defaults
    default_tenant_id = str(_getenv("DEFAULT_TENANT_ID", "local-dev"))
    default_user_id = str(_getenv("DEFAULT_USER_ID", "local-cli"))
    default_conversation_id = str(_getenv("DEFAULT_CONVERSATION_ID", "local-session"))

    # Gateway model config
    gateway_model_id = str(_getenv("GATEWAY_MODEL_ID", "enterprise-agent"))

    # LLM Router
    llm_router_enabled = _as_bool("LLM_ROUTER_ENABLED", True)
    llm_router_confidence_threshold = _as_float("LLM_ROUTER_CONFIDENCE_THRESHOLD", 0.70)

    # Web search fallback
    tavily_api_key = _getenv("TAVILY_API_KEY")
    web_search_enabled = _as_bool("WEB_SEARCH_ENABLED", False)

    # Data Analyst / Sandbox
    sandbox_docker_image = str(_getenv("SANDBOX_DOCKER_IMAGE", "python:3.12-slim"))
    sandbox_timeout_seconds = _as_int("SANDBOX_TIMEOUT_SECONDS", 60)
    sandbox_memory_limit = str(_getenv("SANDBOX_MEMORY_LIMIT", "512m"))
    data_analyst_max_steps = _as_int("DATA_ANALYST_MAX_STEPS", 10)
    data_analyst_skills_path = Path(_getenv("DATA_ANALYST_SKILLS_PATH", str(skills_dir / "data_analyst_agent.md")))

    # Session Workspace
    workspace_dir = Path(_getenv("WORKSPACE_DIR", str(data_dir / "workspaces")))
    workspace_session_ttl_hours = _as_int("WORKSPACE_SESSION_TTL_HOURS", 24)

    # Ensure backend values are in allowed sets.
    if database_backend not in {"postgres"}:
        raise ValueError(f"Unsupported DATABASE_BACKEND={database_backend!r}. Supported: postgres")
    if vector_store_backend not in {"pgvector"}:
        raise ValueError(f"Unsupported VECTOR_STORE_BACKEND={vector_store_backend!r}. Supported: pgvector")
    if object_store_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported OBJECT_STORE_BACKEND={object_store_backend!r}. Supported: local, s3, azure_blob")
    if skills_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported SKILLS_BACKEND={skills_backend!r}. Supported: local, s3, azure_blob")
    if prompts_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported PROMPTS_BACKEND={prompts_backend!r}. Supported: local, s3, azure_blob")

    # Ensure base local directories exist.
    for p in [data_dir, kb_dir, uploads_dir, skills_dir, prompts_dir, workspace_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return Settings(
        database_backend=database_backend,
        vector_store_backend=vector_store_backend,
        object_store_backend=object_store_backend,
        skills_backend=skills_backend,
        prompts_backend=prompts_backend,
        llm_provider=llm_provider,
        embeddings_provider=embeddings_provider,
        judge_provider=judge_provider,
        ollama_base_url=ollama_base_url,
        ollama_chat_model=ollama_chat_model,
        ollama_embed_model=ollama_embed_model,
        ollama_judge_model=ollama_judge_model,
        ollama_temperature=ollama_temperature,
        ollama_num_predict=ollama_num_predict,
        demo_ollama_num_predict=demo_ollama_num_predict,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_api_version=azure_openai_api_version,
        azure_openai_chat_deployment=azure_openai_chat_deployment,
        azure_openai_judge_deployment=azure_openai_judge_deployment,
        azure_openai_embed_deployment=azure_openai_embed_deployment,
        azure_temperature=azure_temperature,
        nvidia_openai_endpoint=nvidia_openai_endpoint,
        nvidia_api_token=nvidia_api_token,
        nvidia_chat_model=nvidia_chat_model,
        nvidia_judge_model=nvidia_judge_model,
        nvidia_temperature=nvidia_temperature,
        judge_temperature=judge_temperature,
        http2_enabled=http2_enabled,
        ssl_verify=ssl_verify,
        ssl_cert_file=ssl_cert_file,
        tiktoken_enabled=tiktoken_enabled,
        tiktoken_cache_dir=tiktoken_cache_dir,
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
        kb_source_uri=kb_source_uri,
        uploads_source_uri=uploads_source_uri,
        skills_dir=skills_dir,
        prompts_dir=prompts_dir,
        shared_skills_path=shared_skills_path,
        general_agent_skills_path=general_agent_skills_path,
        rag_agent_skills_path=rag_agent_skills_path,
        supervisor_agent_skills_path=supervisor_agent_skills_path,
        utility_agent_skills_path=utility_agent_skills_path,
        basic_chat_skills_path=basic_chat_skills_path,
        judge_grading_prompt_path=judge_grading_prompt_path,
        judge_rewrite_prompt_path=judge_rewrite_prompt_path,
        grounded_answer_prompt_path=grounded_answer_prompt_path,
        rag_synthesis_prompt_path=rag_synthesis_prompt_path,
        parallel_rag_synthesis_prompt_path=parallel_rag_synthesis_prompt_path,
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
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        default_conversation_id=default_conversation_id,
        gateway_model_id=gateway_model_id,
        llm_router_enabled=llm_router_enabled,
        llm_router_confidence_threshold=llm_router_confidence_threshold,
        tavily_api_key=tavily_api_key,
        web_search_enabled=web_search_enabled,
        sandbox_docker_image=sandbox_docker_image,
        sandbox_timeout_seconds=sandbox_timeout_seconds,
        sandbox_memory_limit=sandbox_memory_limit,
        data_analyst_max_steps=data_analyst_max_steps,
        data_analyst_skills_path=data_analyst_skills_path,
        workspace_dir=workspace_dir,
        workspace_session_ttl_hours=workspace_session_ttl_hours,
    )
