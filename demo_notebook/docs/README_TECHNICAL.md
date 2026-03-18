# Demo Notebook Technical Guide

This technical guide covers the standalone `demo_notebook` runtime.
It is independent from the production application and does not import `src/agentic_chatbot`.

## 1) Architecture at a glance

Runtime package: `demo_notebook/runtime`

- `config.py`: loads notebook-local `.env` settings.
- `providers.py`: builds provider clients for `azure`, `ollama`, `vllm`, or `nvidia`.
- `stores.py`: pgvector-backed notebook schema (`dn_documents`, `dn_chunks`).
- `ingest.py`: KB indexing/chunking pipeline for `data/kb`.
- `tools.py`: calculator + document/RAG tools used by agents.
- `router.py`: deterministic BASIC vs AGENT routing.
- `supervisor.py`: supervisor node for specialist selection.
- `graph_builder.py`: LangGraph multi-agent graph (supervisor, utility, RAG, parallel workers).
- `rag_agent.py`: specialist RAG agent.
- `general_agent.py`: general ReAct agent path.
- `orchestrator.py`: top-level entry point (`DemoOrchestrator.process_turn`).
- `observability.py`: print trace callback (`[NOTEBOOK]`, `[ROUTER]`, `[GRAPH]`).
- `skills.py`: prompt composition from local skill files (showcase mode only).

## 2) Prerequisites

- Python 3.11+
- PostgreSQL with `pgvector` extension available
- Access to one provider mode:
  - Azure OpenAI
  - Ollama
  - vLLM OpenAI-compatible endpoint
  - NVIDIA OpenAI-compatible endpoint

## 3) Setup and run

From `/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/check_isolation.py
jupyter notebook agentic_rag_showcase.ipynb
```

## 4) Provider configuration

Set `NOTEBOOK_PROVIDER` in `.env`:

- `azure`: requires API key, endpoint, and deployment names.
- `ollama`: requires base URL + chat/judge/embed model names.
- `vllm`: requires OpenAI-compatible base URL and chat model; embeddings can use endpoint or local fallback.
- `nvidia`: requires OpenAI-compatible endpoint + bearer token + chat/judge model names; embeddings backend is selected separately via `NOTEBOOK_NVIDIA_EMBEDDINGS_BACKEND`.

Key vars are documented in `/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/.env.example`.

### Corporate TLS note

For Azure/vLLM SSL interception or private CA environments:
- `NOTEBOOK_HTTP2=true`
- `NOTEBOOK_HTTPX_TRUST_ENV=true` (uses proxy env vars if set)
- `NOTEBOOK_SSL_VERIFY=false` (common corporate fallback) or `true` with CA bundle
- `NOTEBOOK_SSL_CERT_FILE=/absolute/path/to/company-ca.pem`
- `NOTEBOOK_TIKTOKEN_CACHE_DIR=./.cache/tiktoken` (avoid runtime encoding download failures)

If your security policy allows certificate validation, prefer `NOTEBOOK_SSL_VERIFY=true` with `NOTEBOOK_SSL_CERT_FILE`.
For embedding paths only: `NOTEBOOK_TIKTOKEN_ENABLED=false` bypasses tiktoken tokenization.

### Proxy vs TLS clarification

- `NOTEBOOK_SSL_VERIFY=false` disables certificate verification only.
- It does **not** disable proxy routing.
- `NOTEBOOK_HTTPX_TRUST_ENV` controls whether `httpx` honors `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`.

## 5) Database schema

The notebook runtime uses its own schema objects:

- `dn_documents`
- `dn_chunks`

`dn_chunks.embedding` uses `vector(NOTEBOOK_EMBEDDING_DIM)`.
If you change embedding dimension, drop and recreate `dn_documents`/`dn_chunks` for a clean reindex.

## 6) Notebook execution flow

Notebook sections:

- A) BASIC route (deterministic direct answer)
- B) AGENT RAG route with evidence citations
- C) Parallel multi-doc orchestration
- D) Explicit GeneralAgent path
- E) Provider switching notes
- F) Skills showcase (baseline vs skills-enabled prompts)

## 7) Troubleshooting

- "No module named ...": install `demo_notebook/requirements.txt` in the active notebook venv.
- "connection refused" to Postgres: check `NOTEBOOK_PG_DSN` host/port and DB availability.
- Empty KB results: confirm `NOTEBOOK_KB_DIR` points to existing files and rerun bootstrap cell.
- `expected 1536 dimensions, not 768`: set `NOTEBOOK_EMBEDDING_DIM=768` for Ollama `nomic-embed-text`, drop `dn_chunks`/`dn_documents`, then rerun bootstrap.
- Prompt behavior not changing in skills demo: verify `NOTEBOOK_SKILLS_ENABLED=true` and run section F cells that force showcase mode.
- Proxy connection errors on NVIDIA chat:
  - try `NOTEBOOK_HTTPX_TRUST_ENV=false` first
  - if Azure needs proxy but NVIDIA should bypass it, set `NOTEBOOK_HTTPX_TRUST_ENV=true` and `NO_PROXY=<nvidia-host>` (plus lowercase `no_proxy`)

## 8) Isolation checks

Run:

```bash
python scripts/check_isolation.py
```

This fails if runtime files import `agentic_chatbot.*`.
