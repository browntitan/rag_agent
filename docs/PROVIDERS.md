# Providers and Backend Config

## Provider Matrix

Supported today:

- chat LLM: `ollama`, `azure`, or `nvidia`
- judge LLM: `ollama`, `azure`, or `nvidia`
- embeddings: `ollama` or `azure`

Azure is the default demo path in `.env.example`.

## Azure OpenAI (Default Demo Path)

```bash
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=azure

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.us/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002

# text-embedding-ada-002 requires 1536-dim vectors
EMBEDDING_DIM=1536

# HTTP/TLS controls for corporate cert/proxy environments
HTTP2_ENABLED=true
SSL_VERIFY=true
SSL_CERT_FILE=/absolute/path/to/company-ca.pem

# tiktoken controls (used by Azure embeddings token-length checks)
TIKTOKEN_ENABLED=true
TIKTOKEN_CACHE_DIR=./data/cache/tiktoken
```

Notes:

- Gov endpoints are supported (`https://<resource>.openai.azure.us`).
- Commercial endpoints are also valid (`https://<resource>.openai.azure.com`).
- Backward-compatible aliases still load: `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_EMBED_DEPLOYMENT`.
- Main app uses `httpx` client wiring for Azure provider calls.
- `SSL_CERT_FILE` is propagated to `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` envs for non-httpx paths.
- If SSL inspection blocks tiktoken downloads, set `TIKTOKEN_ENABLED=false` (or pre-seed `TIKTOKEN_CACHE_DIR`).

If `EMBEDDING_DIM` does not match the DB vector column, run:

```bash
python run.py doctor
python run.py migrate-embedding-dim --yes
```

## Ollama (Optional)

```bash
LLM_PROVIDER=ollama
JUDGE_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_JUDGE_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.2
JUDGE_TEMPERATURE=0.0
EMBEDDING_DIM=768
```

## NVIDIA OpenAI-Compatible Endpoint (Chat/Judge Only)

```bash
LLM_PROVIDER=nvidia
JUDGE_PROVIDER=nvidia
EMBEDDINGS_PROVIDER=ollama   # or azure

NVIDIA_OPENAI_ENDPOINT=https://<your-nvidia-endpoint>/v1
NVIDIA_API_TOKEN=<bearer-token>
NVIDIA_CHAT_MODEL=openaigpt-oss-120b
NVIDIA_JUDGE_MODEL=openaigpt-oss-120b
NVIDIA_TEMPERATURE=0.0

HTTP2_ENABLED=true
SSL_VERIFY=false
```

Notes:

- Authentication is sent as `Authorization: Bearer <token>`.
- `NVIDIA_API_TOKEN` is preferred; legacy `Token` env var is also accepted.
- Endpoint is normalized to include `/v1` if omitted.
- Embeddings remain `ollama|azure` in v1. `EMBEDDINGS_PROVIDER=nvidia` is intentionally rejected by config validation.

## GGUF with Ollama

### Manual (recommended)

1. Put your `.gguf` file and `Modelfile` under `./data/ollama/gguf`.
2. Create a model in the running Ollama container:

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama create my-gguf-model -f /gguf/Modelfile
```

3. Point app settings to the created model:

```bash
OLLAMA_CHAT_MODEL=my-gguf-model
OLLAMA_JUDGE_MODEL=my-gguf-model
```

### Auto-import (explicit opt-in)

```bash
OLLAMA_GGUF_AUTO_IMPORT=true
OLLAMA_GGUF_MODEL_NAME=my-gguf-model
OLLAMA_GGUF_MODELFILE=/gguf/Modelfile
```

Run importer only when needed:

```bash
docker compose --profile ollama --profile ollama-import up ollama-gguf-importer
```

This importer is not part of default compose startup and does not run on image build.

## Storage / Backend Switches

```bash
DATABASE_BACKEND=postgres
VECTOR_STORE_BACKEND=pgvector
OBJECT_STORE_BACKEND=local
SKILLS_BACKEND=local
PROMPTS_BACKEND=local
```

Current implementation supports `postgres` + `pgvector` + local file-backed skills/prompts/ingestion.

`OBJECT_STORE_BACKEND=s3|azure_blob` and remote skills/prompts backends are scaffolded in config but not implemented yet.

## Path-Based Prompt / Skills Config

```bash
SKILLS_DIR=./data/skills
PROMPTS_DIR=./data/prompts

SHARED_SKILLS_PATH=./data/skills/skills.md
GENERAL_AGENT_SKILLS_PATH=./data/skills/general_agent.md
RAG_AGENT_SKILLS_PATH=./data/skills/rag_agent.md
SUPERVISOR_AGENT_SKILLS_PATH=./data/skills/supervisor_agent.md
UTILITY_AGENT_SKILLS_PATH=./data/skills/utility_agent.md
BASIC_CHAT_SKILLS_PATH=./data/skills/basic_chat.md

JUDGE_GRADING_PROMPT_PATH=./data/prompts/judge_grading.txt
JUDGE_REWRITE_PROMPT_PATH=./data/prompts/judge_rewrite.txt
GROUNDED_ANSWER_PROMPT_PATH=./data/prompts/grounded_answer.txt
RAG_SYNTHESIS_PROMPT_PATH=./data/prompts/rag_synthesis.txt
PARALLEL_RAG_SYNTHESIS_PROMPT_PATH=./data/prompts/parallel_rag_synthesis.txt
```

## GraphRAG LLM Config (opt-in)

Microsoft GraphRAG uses its own LLM settings, separate from the main app providers. These are written to a per-document `settings.yaml` by `graphrag/config.py`. By default GraphRAG is wired to Ollama — no API key required.

```bash
# Ollama (default — no API key needed)
GRAPHRAG_COMPLETION_MODEL=ollama/qwen3.5:9b        # entity extraction + community summarization
GRAPHRAG_EMBEDDING_MODEL=ollama/nomic-embed-text:latest  # chunk embeddings for the graph index
GRAPHRAG_OLLAMA_BASE_URL=http://localhost:11434     # Ollama server address

# Optional: switch to a cloud provider
# GRAPHRAG_COMPLETION_MODEL=gpt-4.1-mini
# GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
# GRAPHRAG_API_KEY=<your-openai-key>
# GRAPHRAG_EMBEDDING_API_KEY=<your-openai-key>
```

The `generate_graphrag_settings()` function auto-detects whether each model string starts with `ollama/` and writes `api_base` + a dummy `"ollama"` API key (required by LiteLLM). For cloud models it falls back to `${GRAPHRAG_API_KEY}` / `${GRAPHRAG_EMBEDDING_API_KEY}`.

GraphRAG does not share the app's `LLM_PROVIDER` / `JUDGE_PROVIDER` config. It runs `graphrag index` and `graphrag query` as CLI subprocesses — no Docker required.
