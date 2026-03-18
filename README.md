# Agentic RAG Chatbot v2

A production-grade document intelligence chatbot built on LangChain/LangGraph. It combines a deterministic router, a multi-agent supervisor graph (RAG + utility specialists), and a legacy single-agent fallback path, all backed by PostgreSQL + pgvector. The system can reason across multiple documents, extract and compare clauses, identify requirements, and produce grounded answers with inline citations.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [Container Setup](#4-container-setup)
5. [Installation](#5-installation)
6. [Configuration](#6-configuration)
7. [Database Setup](#7-database-setup)
8. [First Run](#8-first-run)
9. [CLI Reference](#9-cli-reference)
10. [Adding Documents](#10-adding-documents)
11. [Customising Agent Behaviour — skills.md](#11-customising-agent-behaviour--skillsmd)
12. [Technical Architecture](#12-technical-architecture)
13. [End-to-End Workflow](#13-end-to-end-workflow)
14. [Example Queries](#14-example-queries)
15. [Limitations](#15-limitations)
16. [Troubleshooting](#16-troubleshooting)
17. [Environment Variable Reference](#17-environment-variable-reference)
18. [Project Layout](#18-project-layout)

---

## 1. What This System Does

This chatbot answers questions by autonomously deciding whether a query needs document search or can be answered directly, then executing a multi-step tool loop to retrieve, compare, and synthesise evidence from indexed documents.

**Core capabilities:**

| Capability | Example Query |
|---|---|
| General knowledge | "What is fan-out in agentic systems?" |
| Document Q&A | "What does our internal policy say about data retention?" |
| Clause extraction | "What does clause 33 say in the supply chain agreement?" |
| Requirements extraction | "Find all requirements from the specification document" |
| Document diff | "What are the differences between contract_v1 and contract_v2?" |
| Clause-by-clause comparison | "Go through both termsets and compare their clauses one by one" |
| Sequential document processing | "First read doc_1, then answer the questions in doc_2" |
| Math | "What is 15% of £2,340,000?" |
| Persistent memory | "Remember that the contract value is £2.3M" |

**Document types supported:**

- Plain text (`.txt`, `.md`)
- PDF — native text extraction + automatic OCR fallback for scanned pages
- Word documents (`.docx`)
- Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`) via PaddleOCR

---

## 2. Architecture Overview

```
User Input
   │
   ▼
Orchestrator (ChatbotApp.process_turn)
   │
   ▼
Deterministic Router (BASIC | AGENT)
   │
   ├─ BASIC -> Basic Chat (LLM only, no tools)
   │
   └─ AGENT (primary) -> Multi-Agent Supervisor Graph
                        ├─ supervisor -> rag_agent -> supervisor
                        ├─ supervisor -> utility_agent -> supervisor
                        ├─ supervisor -> parallel_planner
                        │              -> rag_worker x N
                        │              -> rag_synthesizer
                        │              -> supervisor
                        └─ supervisor -> __end__

AGENT fallback (capability/config issue):
   -> GeneralAgent (legacy) with tools:
      calculator, list_indexed_docs, memory_*, rag_agent_tool

Upload path:
   ingest_paths() -> direct run_rag_agent() summary kickoff

Storage:
   PostgreSQL + pgvector + pg_trgm
   tables: documents, chunks, memory
```

Detailed C4 diagrams (including Level 3 runtime components) are in
[`docs/C4_ARCHITECTURE.md`](docs/C4_ARCHITECTURE.md#c4-level-3-component-view-application-runtime-container).

---

## 3. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| Docker | 24+ | For PostgreSQL and optional Ollama containers |
| Ollama **or** Azure OpenAI | — | One is required |
| PostgreSQL 15/16 with pgvector | — | Easiest via Docker (see below) |
| LangGraph | `>=0.2.0` | Installed automatically via `requirements.txt` |
| 8 GB RAM | — | 16 GB recommended for larger models |
| Internet access | — | PaddleOCR downloads models on first use (~200 MB) |

**GPU (optional):** Required only if you set `OCR_USE_GPU=true`. The system runs fine on CPU.

---

## 4. Container Setup

### 4.0 One-Command Full Stack (Recommended)

This repo now includes:

- `Dockerfile` for the app container
- `docker-compose.yml` for app + pgvector DB + optional Ollama + optional Langfuse stack

Quick start:

```bash
cp .env.example .env

# Azure-first stack: app gateway + pgvector
docker compose up -d --build

# Optional: include Ollama runtime locally
docker compose --profile ollama up -d --build

# Optional observability stack (Langfuse + deps)
docker compose --profile observability up -d --build
```

This works even if PostgreSQL is not installed on your machine; `rag-postgres` runs inside Docker and the app connects to it automatically.

Ollama models are never pulled/created automatically during image build or container startup. If using local Ollama, pull/create models manually:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen3:8b
```

The app container now starts the OpenAI-compatible gateway on boot (`http://localhost:8000`). You can still run CLI commands inside the app container:

```bash
# schema migration + KB indexing
docker compose exec app python run.py migrate
docker compose exec app python run.py init-kb

# interactive app
docker compose exec app python run.py chat

# one-shot/demo
docker compose exec app python run.py ask -q "What does the API auth doc say?"
docker compose exec app python run.py demo --list-scenarios
```

If you need one-shot GGUF auto-import, it is explicit opt-in via profile `ollama-import` and env flags.

### 4.1 PostgreSQL with pgvector (Required)

The database stores document chunks, vector embeddings, and cross-turn memory.

```bash
docker pull pgvector/pgvector:pg16

docker run -d \
  --name ragdb \
  -e POSTGRES_DB=ragdb \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpass \
  -p 5432:5432 \
  --restart unless-stopped \
  pgvector/pgvector:pg16
```

Verify it is running:

```bash
docker exec ragdb psql -U raguser -d ragdb -c "SELECT version();"
```

The application will create its own tables via the `migrate` command — no manual SQL needed.

**Connection string** for your `.env`:
```
PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
```

If you run the app itself via Docker Compose, this host-based `PG_DSN` is overridden internally to:

```env
postgresql://raguser:ragpass@rag-postgres:5432/ragdb
```

### 4.2 Ollama (Optional)

Ollama serves the local LLM and embedding model.

```bash
# Pull and start the Ollama container
docker pull ollama/ollama

docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama
```

Pull the required models inside the container:

```bash
# Embedding model (required — 768-dim output)
docker exec ollama ollama pull nomic-embed-text

# Chat model (choose one)
docker exec ollama ollama pull qwen3:8b          # project default / fastest startup
docker exec ollama ollama pull qwen2.5:14b       # stronger quality
docker exec ollama ollama pull llama3.1:8b       # alternative
docker exec ollama ollama pull llama3.1:70b      # high quality, high resource usage
```

Use your own GGUF model with Ollama (manual flow):

1. Put `.gguf` and `Modelfile` in `./data/ollama/gguf/`
2. Create model inside the running Ollama container
3. Point app config to that model name

```bash
docker compose exec ollama ollama create my-gguf-model -f /gguf/Modelfile
```

```env
OLLAMA_CHAT_MODEL=my-gguf-model
OLLAMA_JUDGE_MODEL=my-gguf-model
```

Optional compose auto-import (one-shot helper service):

```env
OLLAMA_GGUF_AUTO_IMPORT=true
OLLAMA_GGUF_MODEL_NAME=my-gguf-model
OLLAMA_GGUF_MODELFILE=/gguf/Modelfile
```

Run importer explicitly:

```bash
docker compose --profile ollama --profile ollama-import up ollama-gguf-importer
```

Verify Ollama is accessible:

```bash
curl http://localhost:11434/api/tags
```

> **GPU acceleration for Ollama:** Add `--gpus all` to the `docker run` command if you have an NVIDIA GPU and the NVIDIA Container Toolkit installed.

### 4.3 Langfuse Observability (Optional)

Langfuse can be used with this codebase today.

Why: the app already wires LangChain/LangGraph callbacks through `get_langchain_callbacks(...)` in `src/agentic_chatbot/observability/callbacks.py` and emits traces whenever `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.

Start the built-in Langfuse stack from this repo:

```bash
docker compose --profile observability up -d
```

Open Langfuse:

- UI: `http://localhost:3000`
- MinIO console (optional): `http://localhost:${LANGFUSE_MINIO_CONSOLE_PORT:-9091}`

Then either:

1. Create a project in the UI and copy keys into `.env`, or
2. Pre-seed project/user via `LANGFUSE_INIT_*` variables in `.env` before starting compose.

Set/verify these in `.env` for the app:

```env
# if running app on host:
LANGFUSE_HOST=http://localhost:3000

# if running app via docker compose:
LANGFUSE_HOST_DOCKER=http://langfuse-web:3000

LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_DEBUG=false
```

Restart app container after key changes:

```bash
docker compose restart app
```

### 4.4 OpenAI-Compatible Agent Gateway

If you started Docker Compose, the gateway is already running on `http://localhost:8000`.

To run on host instead:

```bash
python run.py serve-api --host 0.0.0.0 --port 8000
```

Endpoint summary:

- `GET http://localhost:8000/v1/models`
- `POST http://localhost:8000/v1/chat/completions`
- `POST http://localhost:8000/v1/ingest/documents`

The gateway is designed so OpenWebUI and AI SDK can target it as an OpenAI-compatible backend by changing base URL and credentials.

In simplified mode, the gateway has no built-in auth. For production, place it behind a trusted network boundary or upstream proxy/API gateway.

OpenWebUI wiring:

1. Provider type: OpenAI-compatible.
2. Base URL: `http://<gateway-host>:8000/v1`.
3. Model: `enterprise-agent` (or `GATEWAY_MODEL_ID`).
4. Optionally set `X-Conversation-ID` for stable chat thread memory scope.

AI SDK wiring:

1. Point your provider/base URL to `http://<gateway-host>:8000/v1`.
2. Use model `enterprise-agent`.
3. Optionally pass `X-Conversation-ID: <stable-chat-id>` for stable chat thread memory scope.

Example request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-ID: demo-chat-001" \
  -d '{
    "model": "enterprise-agent",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Compare api_auth.md and api_examples.md with citations."}
    ],
    "stream": false
  }'
```

---

## 5. Installation

### 5.1 Clone and Set Up Python Environment

```bash
git clone <repository-url>
cd langchain_agentic_chatbot_v2

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 5.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PaddleOCR:** The first time a PDF or image file triggers OCR, PaddleOCR downloads its models (~200 MB). This requires a one-time internet connection. Subsequent runs use the cached models.

> **GPU users:** Replace `paddlepaddle` with `paddlepaddle-gpu` in `requirements.txt` before installing, then set `OCR_USE_GPU=true` in your `.env`.

### 5.3 Verify Installation

```bash
python run.py --help
```

Expected output:
```
Usage: run.py [OPTIONS] COMMAND [ARGS]...

Commands:
  ask            Run a single-turn query.
  chat           Start an interactive chat session.
  init-kb        Force (re)indexing of the built-in demo KB.
  migrate        Apply the database schema (idempotent).
  reset-indexes  Truncate all indexed data from PostgreSQL.
  demo           Run curated multi-turn demo scenarios.
  serve-api      Run the OpenAI-compatible FastAPI gateway.
```

---

## 6. Configuration

### 6.1 Create Your `.env` File

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
# ── Backends (current + future-ready switches) ───────────────────
DATABASE_BACKEND=postgres
VECTOR_STORE_BACKEND=pgvector
OBJECT_STORE_BACKEND=local
SKILLS_BACKEND=local
PROMPTS_BACKEND=local

# ── Provider (Azure-first demo defaults) ──────────────────────────
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=azure

# ── Azure OpenAI (Gov endpoints supported) ───────────────────────
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.us/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002

# Corporate cert / SSL interception controls
HTTP2_ENABLED=true
SSL_VERIFY=true
SSL_CERT_FILE=/absolute/path/to/company-ca.pem

# If tiktoken download is blocked by SSL policy:
TIKTOKEN_ENABLED=false

# ── Database ──────────────────────────────────────────────────────
PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
EMBEDDING_DIM=1536                    # required for text-embedding-ada-002
```

Legacy aliases are still accepted: `AZURE_OPENAI_DEPLOYMENT` (chat) and `AZURE_OPENAI_EMBED_DEPLOYMENT` (embeddings).

If you run backend commands in the `app` container, Compose injects an internal `PG_DSN` to the `rag-postgres` service automatically.

### 6.2 Using NVIDIA Endpoint Instead of Azure (Optional)

```env
LLM_PROVIDER=nvidia
JUDGE_PROVIDER=nvidia
EMBEDDINGS_PROVIDER=ollama   # or azure

NVIDIA_OPENAI_ENDPOINT=https://openaigpt-oss-120b-lighthouse-ai-dev-vllm.apps.lh-prod.ekho.myngc.com/v1
NVIDIA_API_TOKEN=<your-token>
NVIDIA_CHAT_MODEL=openaigpt-oss-120b
NVIDIA_JUDGE_MODEL=openaigpt-oss-120b
NVIDIA_TEMPERATURE=0.0

HTTP2_ENABLED=true
SSL_VERIFY=false
TIKTOKEN_ENABLED=false
```

Validate config quickly:

```bash
python run.py doctor --skip-db --skip-ollama
```

Troubleshooting:
- `401/403`: verify `NVIDIA_API_TOKEN`.
- `404 model not found`: check `NVIDIA_CHAT_MODEL` / `NVIDIA_JUDGE_MODEL`.
- SSL issues: keep `SSL_VERIFY=false` or set `SSL_VERIFY=true` + `SSL_CERT_FILE=/path/to/ca.pem`.

### 6.3 Using Ollama Instead of Azure (Optional)

```env
LLM_PROVIDER=ollama
JUDGE_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_JUDGE_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_NUM_PREDICT=2048
DEMO_OLLAMA_NUM_PREDICT=2048

# nomic-embed-text outputs 768 dims:
EMBEDDING_DIM=768
```

> **Important:** If you change embedding models/dimensions on an existing DB, run:
> `python run.py migrate-embedding-dim --yes`

### 6.4 Skills and Prompt Template Paths

All system prompts and judge/synthesis prompt templates are path-configurable:

```env
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

---

## 7. Database Setup

Apply the schema (creates tables, indexes, and PostgreSQL extensions):

```bash
python run.py migrate
```

This command is idempotent — safe to run multiple times. It will not delete existing data.

**What is created:**

| Object | Type | Purpose |
|---|---|---|
| `vector` extension | Extension | pgvector for ANN vector search |
| `pg_trgm` extension | Extension | Trigram-based fuzzy title matching |
| `documents` table | Table | Document metadata (title, type, hash) |
| `chunks` table | Table | Document chunks with embeddings and full-text index |
| `memory` table | Table | Cross-turn session memory |
| HNSW index | Index | Fast approximate nearest-neighbour search on embeddings |
| GIN index | Index | Full-text search on chunk content |

### 7.1 Database Setup via Docker Compose

If you started the containerized stack, run migrations from the app container:

```bash
docker compose exec app python run.py migrate
docker compose exec app python run.py init-kb
```

The app entrypoint can also auto-run migrations (`APP_AUTO_MIGRATE=true`), but running the explicit commands above is the most reliable first-run check.

### 7.2 Backend Bootstrap Checklist (End-to-End)

Run these in order on a fresh setup:

```bash
# 1) start PostgreSQL (pgvector image) and Ollama/Azure config
# 2) install deps and create .env

python run.py doctor
python run.py migrate
python run.py init-kb

# sanity check: single question
python run.py ask -q "What authentication methods does the API support?"

# interactive backend
python run.py chat

# curated demo suite
python run.py demo --list-scenarios
python run.py demo --scenario all --max-turns 2
```

---

## 8. First Run

### 8.0 Run Preflight Diagnostics

```bash
python run.py doctor
```

This checks provider package imports, DB connectivity, and Ollama reachability (when Ollama is selected).

### 8.1 Index the Built-in Knowledge Base

The `data/kb/` directory contains 18 demo documents (product docs, API references, runbooks, and long-form enterprise contract/compliance packs). Index them:

```bash
python run.py init-kb
```

This is also done automatically the first time you run `ask` or `chat`.

### 8.2 Ask a Single Question

```bash
python run.py ask -q "What authentication methods does the API support?"
```

### 8.3 Start an Interactive Session

```bash
python run.py chat
```

Type your question at the `You>` prompt. Use `/upload PATH` to ingest a document mid-conversation, or `/exit` to quit.

### 8.4 Run Curated Demo Scenarios

```bash
python run.py demo --list-scenarios
python run.py demo --scenario utility_memory_finance_bootstrap --verify
python run.py demo --scenario parallel_rag_multi_doc_risk_board --force-agent --verify
python run.py demo --scenario all --session-mode scenario --verify
```

`demo` mode applies two reliability guards by default:
- raises `OLLAMA_NUM_PREDICT` to `DEMO_OLLAMA_NUM_PREDICT` (if higher)
- compacts `list_indexed_docs` tool output to grouped categories (demo sessions only)

Manager-facing showcase order (recommended):

1. `utility_memory_finance_bootstrap`
2. `rag_clause_navigation_and_extraction`
3. `rag_structural_diff_contract_versions`
4. `parallel_rag_multi_doc_risk_board`
5. `executive_due_diligence_grand_finale`

If Langfuse is enabled, run demo scenarios and inspect traces in `http://localhost:3000` to show:

- router decisions
- supervisor handoffs
- multi-step tool usage
- final synthesis traces

### 8.5 Demo Showcase Catalog

| Scenario ID | Primary Objective | Core Tool/Flow Coverage |
|---|---|---|
| `utility_memory_finance_bootstrap` | Prove deterministic utility behavior | `calculator`, `list_indexed_docs`, `memory_*` |
| `rag_resolution_and_search_strategy` | Show resolution + retrieval strategy switching | `resolve_document`, `search_document`, `search_all_documents` |
| `rag_clause_navigation_and_extraction` | Extract exact legal clauses | `list_document_structure`, `extract_clauses` |
| `rag_requirements_traceability` | Build requirements trace map | `extract_requirements`, scoped search |
| `rag_structural_diff_contract_versions` | Surface structural deltas between versions | `diff_documents`, synthesis |
| `rag_clause_compare_conflict_review` | Detect clause conflicts and strictness shifts | `compare_clauses`, scratchpad tools |
| `parallel_rag_multi_doc_risk_board` | Demonstrate fan-out/fan-in orchestration | supervisor `parallel_rag`, workers, synthesizer |
| `executive_due_diligence_grand_finale` | End-to-end board-ready recommendation | mixed utility + RAG + memory |

---

## 9. CLI Reference

### `ask` — Single-Turn Query

```bash
python run.py ask [OPTIONS]
```

| Option | Description |
|---|---|
| `-q TEXT` / `--question TEXT` | The question to ask (required) |
| `-u PATH` / `--upload PATH` | File to ingest before asking (repeatable) |
| `--force-agent` | Skip router and force AGENT path (supervisor graph; fallback to legacy GeneralAgent if needed) |
| `--dotenv PATH` | Load a specific `.env` file |

**Examples:**

```bash
# Simple question
python run.py ask -q "What is the rate limit for the API?"

# Ingest a PDF and ask about it
python run.py ask -q "Summarise the key clauses" -u ./contract.pdf

# Ingest multiple files
python run.py ask -q "Compare these two documents" \
  -u ./termset_v1.pdf \
  -u ./termset_v2.pdf

# Force agent mode (bypass router)
python run.py ask -q "Hello" --force-agent
```

---

### `chat` — Interactive Session

```bash
python run.py chat [OPTIONS]
```

| Option | Description |
|---|---|
| `-u PATH` / `--upload PATH` | File(s) to ingest at session start (repeatable) |
| `--dotenv PATH` | Load a specific `.env` file |

**In-session commands:**

| Command | Description |
|---|---|
| `/upload PATH` | Ingest a document mid-conversation |
| `/exit` or `/quit` | End the session |

**Example:**

```bash
python run.py chat -u ./policy.pdf -u ./contract.pdf
```

---

### `migrate` — Apply Database Schema

```bash
python run.py migrate [--dotenv PATH]
```

Idempotent. Run this after initial setup and after any `schema.sql` changes.

---

### `migrate-embedding-dim` — Realign Vector Dimension + Reindex

```bash
python run.py migrate-embedding-dim [--yes] [--target-dim N] [--reindex-kb/--skip-reindex-kb] [--reset-memory/--keep-memory] [--dotenv PATH]
```

Use this when switching embedding models (for example, `nomic-embed-text` 768 -> `text-embedding-ada-002` 1536).  
It aligns `chunks.embedding` to the target dimension, clears indexed docs/chunks, and optionally rebuilds KB vectors.

---

### `doctor` — Preflight Providers + Connectivity

```bash
python run.py doctor [OPTIONS]
```

| Option | Description |
|---|---|
| `--dotenv PATH` | Load a specific `.env` file |
| `--strict` | Exit non-zero if warnings are present |
| `--timeout-seconds FLOAT` | Timeout for connectivity checks (default: `3.0`) |
| `--check-db / --skip-db` | Enable or skip PostgreSQL connectivity check |
| `--check-ollama / --skip-ollama` | Enable or skip Ollama API check when Ollama providers are selected |

Examples:

```bash
python run.py doctor
python run.py doctor --strict
python run.py doctor --skip-ollama
```

---

### `init-kb` — Index the Knowledge Base

```bash
python run.py init-kb [--dotenv PATH]
```

Forces re-indexing of all files in `data/kb/`. Documents already indexed with the same content hash are skipped.

---

### `reset-indexes` — Clear All Indexed Data

```bash
python run.py reset-indexes [--yes] [--dotenv PATH]
```

| Option | Description |
|---|---|
| `--yes` / `-y` | Skip the confirmation prompt |

**Warning:** This truncates the `documents`, `chunks`, and `memory` tables. All indexed content and session memory is permanently deleted. Run `init-kb` or `chat` to rebuild.

---

### `demo` — Run Curated Demo Scenarios

```bash
python run.py demo [OPTIONS]
```

| Option | Description |
|---|---|
| `-s TEXT` / `--scenario TEXT` | Scenario name or `all` (default: `all`) |
| `--list-scenarios` | List available scenarios and exit |
| `--max-turns INT` | Max prompts per scenario (`0` = all) |
| `--force-agent` | Force AGENT path for all demo prompts |
| `--session-mode TEXT` | `scenario` (fresh session per scenario) or `suite` (shared session) |
| `--verify` | Run heuristic checks and print `PASS/WARN/FAIL` per turn |
| `--show-notes` | Print scenario briefing notes before execution |
| `-u PATH` / `--upload PATH` | Ingest file(s) before demo starts |
| `--continue-on-error / --stop-on-error` | Continue or abort on first failing prompt |
| `--dotenv PATH` | Load a specific `.env` file |

Examples:

```bash
python run.py demo --list-scenarios
python run.py demo --scenario utility_memory_finance_bootstrap --verify
python run.py demo --scenario rag_clause_compare_conflict_review --force-agent --verify
python run.py demo --scenario all --session-mode scenario --max-turns 2
```

---

### `serve-api` — Run OpenAI-Compatible Gateway

```bash
python run.py serve-api [--host 0.0.0.0] [--port 8000] [--reload]
```

This starts a FastAPI server exposing:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/ingest/documents`

---

## 10. Adding Documents

### 10.1 Knowledge Base Documents (Permanent)

Place files in `data/kb/` and run:

```bash
python run.py init-kb
```

KB documents are indexed once and persist across sessions. They are tagged `source_type='kb'`.

### 10.2 User Uploads (Session Uploads)

Upload documents during a session. They are ingested, embedded, and summarised automatically:

```bash
# At startup
python run.py chat -u ./my_document.pdf

# Mid-session (in the chat REPL)
/upload ./my_document.pdf

# One-shot upload + question
python run.py ask -q "Find all requirements" -u ./spec.pdf
```

Uploads are tagged `source_type='upload'` and persist in the database across sessions.

### 10.3 Supported File Types

| Extension | Method | Notes |
|---|---|---|
| `.txt`, `.md` | TextLoader | UTF-8 encoding |
| `.pdf` | PyPDF + PaddleOCR | Native text first; OCR fallback for pages with < 50 chars |
| `.docx` | Docx2txtLoader | Text extraction only (no images within DOCX) |
| `.png`, `.jpg`, `.jpeg` | PaddleOCR | Full OCR with reading-order preservation |
| `.tiff`, `.tif`, `.bmp`, `.gif` | PaddleOCR | Full OCR |

**Controlling OCR:**

```env
USE_PADDLE_OCR=true           # enable/disable (default: true)
OCR_LANGUAGE=en               # language code (en, ch, fr, de, etc.)
OCR_USE_GPU=false             # use GPU for OCR (default: false)
OCR_MIN_PAGE_CHARS=50         # chars below this threshold trigger OCR on a PDF page
```

### 10.4 Document Structure Detection

At ingest time, the system automatically classifies each document into one of five structure types using regex heuristics (no LLM call):

| Type | Detection Signal | Splitting Strategy |
|---|---|---|
| `general` | No clauses or requirements | Generic recursive character splitting |
| `structured_clauses` | Numbered clause headers (Clause N, Section N, Article N) | Clause-boundary splitting |
| `requirements_doc` | shall/must language or REQ-NNN identifiers | Clause or generic + requirement tagging |
| `policy_doc` | Clause headers + policy/compliance keywords | Clause-boundary splitting |
| `contract` | Clause headers + contract/agreement keywords | Clause-boundary splitting |

This classification is stored in the `documents` table and affects how the RAG agent retrieves and compares content.

---

## 11. Customising Agent Behaviour — skills.md

The agents' system prompts are loaded from Markdown files at runtime. You can change agent behaviour without touching Python code.

```
data/skills/
├── skills.md              ← Shared context injected into ALL agents
├── general_agent.md       ← GeneralAgent + BasicChat system prompt (fallback path)
├── rag_agent.md           ← RAGAgent system prompt with tool decision trees
├── supervisor_agent.md    ← Supervisor routing rules (multi-agent graph)
├── utility_agent.md       ← Utility agent instructions (calc, memory, list_docs)
└── basic_chat.md          ← (optional) Dedicated BasicChat prompt
```

### Hot-Reload Behaviour

| File | Loaded When | Hot-Reloadable? |
|---|---|---|
| `data/skills/general_agent.md` | Once at `ChatbotApp` startup | **No** — restart required |
| `data/skills/rag_agent.md` | On every `run_rag_agent()` call | **Yes** — edit and the next RAG query picks it up |
| `data/skills/supervisor_agent.md` | Once per graph build (per turn) | **Yes** — next AGENT turn picks it up |
| `data/skills/utility_agent.md` | Once per graph build (per turn) | **Yes** — next AGENT turn picks it up |
| `data/skills/skills.md` | Same as the agent it's injected into | Follows its agent |

> **RAG Agent live-editing:** Because `rag_agent.md` is reloaded per-turn, you can refine the RAG agent's decision rules mid-session. The change takes effect on the very next question that triggers RAG.

### How Skills Are Combined

Each agent receives a single concatenated system prompt:

```
load_general_agent_skills()  =  skills.md  +  "---"  +  general_agent.md
load_rag_agent_skills()      =  skills.md  +  "---"  +  rag_agent.md
```

If a `.md` file is missing, the system falls back to a hardcoded Python constant — so the system always works even without the `data/skills/` directory.

### Common Customisations

- **Domain context** — add organisation-specific facts to `skills.md`:
  ```markdown
  ## Organisation Context
  We are a supply chain company. Our termsets follow ISO 22000 and contain 52 numbered clauses.
  Always flag clauses that may conflict with GDPR obligations.
  ```
- **Output format** — change how the GeneralAgent presents answers in `general_agent.md`
- **Search strategy hints** — add document-type-specific rules to `rag_agent.md`
- **Citation format** — adjust inline citation style in `skills.md`

For a full governance and authoring guide, see:

- `docs/SKILLS_PLAYBOOK.md`

---

## 12. Technical Architecture

### 12.0 Multi-Agent Graph (Supervisor Pattern)

The AGENT path uses a **LangGraph supervisor graph** that routes to specialist agents instead of running a single GeneralAgent with all tools. This is built with `StateGraph` and the `Send` API for parallel execution.

```
START → supervisor ──→ rag_agent ──→ supervisor (loop)
                  ├──→ utility_agent ──→ supervisor (loop)
                  ├──→ parallel_planner ──→ [rag_worker × N] ──→ rag_synthesizer ──→ supervisor
                  └──→ END
```

**Supervisor** (`graph/supervisor.py`): An LLM node that reads conversation history and returns a JSON routing decision — which specialist agent to invoke next. Valid targets: `rag_agent`, `utility_agent`, `parallel_rag`, `__end__`. The supervisor loops: after each agent finishes, it decides whether to route to another agent or stop.

**RAG Agent** (`graph/nodes/rag_node.py`): Wraps the existing `run_rag_agent()` as a graph node. All 11 RAG tools are unchanged.

**Utility Agent** (`graph/nodes/utility_node.py`): A `create_react_agent` subgraph with `calculator`, `list_indexed_docs`, and `memory_*` tools.

**Parallel RAG** (`graph/nodes/rag_worker_node.py`): Uses the LangGraph `Send` API to fan out N `rag_worker` nodes in parallel — one per document. Results are merged by `rag_synthesizer` through a reducer that supports parallel append plus explicit post-synthesis clearing.

**Fallback**: If the graph cannot run due capability/config limitations (for example tool-calling incompatibility), the orchestrator falls back to the legacy single-agent path (`run_general_agent` with `rag_agent_tool`). Unexpected graph runtime errors are surfaced explicitly instead of silently masking defects.

### 12.1 Request Routing

The router runs before any LLM call. It uses pure regex and heuristics — no token cost:

```
AGENT route triggered when:
  - attachments present (confidence 1.0)
  - tool-like verbs found (calculate, search, find, compare, retrieve, ...)
  - citation/grounding language (cite, sources, according to, ...)
  - high-stakes topics (medical, legal, contract, financial, compliance, ...)
  - long input > 600 characters
  - --force-agent flag set

BASIC route:
  - everything else (general knowledge, small talk)
```

### 12.2 GeneralAgent (Legacy Fallback Path)

The GeneralAgent is powered by **LangGraph `create_react_agent`** and is used as the legacy fallback path.

```
┌──────────────────┐   tool_calls   ┌────────────────────┐
│   agent node     │ ─────────────► │   tools node       │
│   (LLM invoke)   │ ◄───────────── │   (tool execution) │
└──────────────────┘  ToolMessages  └────────────────────┘
         │
         │  no tool_calls
         ▼
   Final AIMessage returned
```

**How it works:**
1. System prompt loaded from `data/skills/general_agent.md` is prepended as a `SystemMessage`
2. User message is appended as a `HumanMessage`
3. Graph is invoked: `graph.invoke({"messages": msgs}, config={"recursion_limit": N})`
4. The ReAct loop runs automatically — LLM → tool execution → LLM — until no further tool calls
5. The result contains the full updated message history; the last `AIMessage` text is returned

**Budget control:** The recursion limit is computed as `(max(MAX_AGENT_STEPS, MAX_TOOL_CALLS) + 1) × 2 + 1`. This accounts for 2 graph node visits per ReAct cycle (agent + tools) plus a buffer. When the limit is hit, `GraphRecursionError` is caught and a graceful partial response is returned.

**Available tools:** `calculator`, `list_indexed_docs`, `memory_save`, `memory_load`, `memory_list`, `rag_agent_tool`.

**Fallback:** If the LLM does not support `bind_tools()`, a plan-execute fallback generates a JSON plan, executes tools sequentially, and synthesises a final answer. This path does not use LangGraph.

### 12.3 RAGAgent

The RAGAgent is powered by **LangGraph `create_react_agent`** and is primarily invoked as a specialist node in the multi-agent graph (`rag_agent`, `rag_worker`). It is also invoked through `rag_agent_tool` in the legacy fallback path. It operates as an autonomous specialist loop with 11 dedicated tools:

| Tool | Purpose |
|---|---|
| `resolve_document(name_or_hint)` | Fuzzy-match a document name to a `doc_id` using rapidfuzz + pg_trgm |
| `search_document(doc_id, query, strategy)` | Vector/keyword/hybrid search scoped to one document |
| `search_all_documents(query, strategy)` | Cross-document search (no doc filter) |
| `extract_clauses(doc_id, clause_numbers)` | Retrieve exact clause text by number ("3", "3.2", "10.1") |
| `list_document_structure(doc_id)` | Show the clause/section outline of a document |
| `extract_requirements(doc_id, filter)` | SQL `WHERE chunk_type='requirement'` with optional semantic re-rank |
| `compare_clauses(doc_id_1, doc_id_2, clause_numbers)` | Side-by-side clause text from two documents |
| `diff_documents(doc_id_1, doc_id_2)` | Structural outline diff (shared vs. unique clauses) |
| `scratchpad_write(key, value)` | Store intermediate findings in `session.scratchpad` |
| `scratchpad_read(key)` | Retrieve stored findings |
| `scratchpad_list()` | List all scratchpad keys |

**How it works:**
1. System prompt loaded from `data/skills/rag_agent.md` is prepended as a `SystemMessage`
2. A task message containing the `QUERY`, `PREFERRED_DOC_IDS`, and strategy hints is passed as a `HumanMessage`
3. LangGraph runs the same ReAct loop — tool calls → tool results → tool calls — until the agent has sufficient evidence
4. A **final synthesis call** asks the LLM to produce the RAG contract JSON from all accumulated tool results
5. The structured contract dict is returned to the caller (graph node or `rag_agent_tool`)

**Budget control:** Uses formula `(MAX_RAG_AGENT_STEPS + MAX_TOOL_CALLS + 1) × 2 + 1` for the recursion limit. On budget exhaustion, synthesis proceeds with whatever evidence was collected before the limit.

The RAGAgent returns a structured contract dict:
```json
{
  "answer": "...",
  "citations": [{"citation_id": "...", "title": "...", "location": "...", "snippet": "..."}],
  "used_citation_ids": ["..."],
  "confidence": 0.92,
  "retrieval_summary": {
    "query_used": "...",
    "steps": 4,
    "tool_calls_used": 7,
    "tool_call_log": ["resolve_document({...})", "search_document({...})", "..."],
    "citations_found": 8
  },
  "followups": ["What does clause 34 say?"],
  "warnings": []
}
```

### 12.4 Document Ingestion Pipeline

```
File path
    │
    ├─ Compute SHA-1 hash (deduplication key)
    │
    ├─ Check doc_store.document_exists() → skip if unchanged
    │
    ├─ _load_documents(path, settings)
    │     ├── .txt/.md  → TextLoader
    │     ├── .pdf      → PyPDF per page; OCR for pages < 50 chars via PyMuPDF + PaddleOCR
    │     ├── .docx     → Docx2txtLoader
    │     └── image     → PaddleOCR → single Document
    │
    ├─ detect_structure(full_text)
    │     → classifies as general / structured_clauses / requirements_doc / policy_doc / contract
    │
    ├─ _split_with_structure(settings, docs, structure)
    │     ├── has_clauses=True  → clause_split() at heading boundaries
    │     └── has_clauses=False → RecursiveCharacterTextSplitter (chunk_size=900, overlap=150)
    │     └── post-tag chunk_type='requirement' where shall/must/REQ-NNN detected
    │
    ├─ _build_chunk_records() → list[ChunkRecord]
    │     (chunk_id, doc_id, chunk_index, content, chunk_type, page_number,
    │      clause_number, section_title, embedding=None)
    │
    ├─ chunk_store.add_chunks(records)
    │     → generate embeddings via embeddings model
    │     → batch INSERT with pgvector
    │
    └─ doc_store.upsert_document(DocumentRecord)
          → title, source_type, file_type, doc_structure_type, num_chunks, content_hash
```

### 12.5 Retrieval

The RAGAgent's search tools support three strategies:

| Strategy | Method | Best For |
|---|---|---|
| `vector` | Cosine similarity via HNSW index on pgvector | Semantic / conceptual questions |
| `keyword` | PostgreSQL `tsvector`/`tsquery` full-text search | Exact term matching, clause numbers, defined terms |
| `hybrid` | Union of vector + keyword results, deduplicated by highest score | General queries (default) |

All search functions accept an optional `doc_id_filter` which is pushed to the SQL `WHERE` clause for efficient single-document search.

### 12.6 Scratchpad and Persistent Memory

**Scratchpad** (`session.scratchpad: dict[str, str]`):
- Lives in the `ChatSession` object
- Cleared at the end of each turn if `CLEAR_SCRATCHPAD_PER_TURN=true`
- Used by the RAGAgent to accumulate intermediate findings across tool calls within a single turn
- Tools: `scratchpad_write`, `scratchpad_read`, `scratchpad_list`

**Persistent Memory** (PostgreSQL `memory` table):
- Keyed by `(tenant_id, session_id, key)`
- Survives across turns and restarts
- Used by the GeneralAgent via `memory_save`, `memory_load`, `memory_list`
- Intended for facts the user explicitly confirms or asks the agent to remember

### 12.7 Database Schema

```sql
-- Tracks indexed documents
CREATE TABLE documents (
    doc_id             TEXT PRIMARY KEY,        -- stable hash-based ID
    title              TEXT NOT NULL,
    source_type        TEXT NOT NULL,           -- 'kb' | 'upload'
    source_path        TEXT,
    content_hash       TEXT NOT NULL,           -- SHA-1 for dedup
    num_chunks         INTEGER DEFAULT 0,
    ingested_at        TIMESTAMPTZ DEFAULT now(),
    file_type          TEXT,                    -- 'pdf' | 'txt' | 'md' | 'docx'
    doc_structure_type TEXT DEFAULT 'general'   -- classification
);

-- Document chunks with vector embeddings
CREATE TABLE chunks (
    chunk_id       TEXT PRIMARY KEY,            -- "doc_id#chunk0042"
    doc_id         TEXT REFERENCES documents,
    chunk_index    INTEGER NOT NULL,
    page_number    INTEGER,                     -- zero-based page
    clause_number  TEXT,                        -- e.g. "3.2", "10.1.4"
    section_title  TEXT,
    content        TEXT NOT NULL,
    embedding      vector(<EMBEDDING_DIM>),     -- HNSW cosine index
    ts             tsvector GENERATED ALWAYS    -- GIN full-text index
                   AS (to_tsvector('english', content)) STORED,
    chunk_type     TEXT DEFAULT 'general'       -- 'general'|'clause'|'requirement'|...
);

-- Cross-turn session memory
CREATE TABLE memory (
    id          SERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'local-dev',
    session_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(tenant_id, session_id, key)
);
```

### 12.8 Observability

When Langfuse keys are configured, every turn emits traces including:
- Router decision and confidence score
- Supervisor graph nodes: routing loops, utility agent runs, RAG node/worker runs, and synthesis steps
- GeneralAgent fallback path (when used): each LLM call, tool call, and result
- Upload ingestion events

Access the Langfuse dashboard at `http://localhost:3000` (if running locally).

---

## 13. End-to-End Workflow

The following traces a query from user input to response:

```
User: "What are the differences between termset_v1 and termset_v2
       in clauses 10 through 15?"

1. Orchestrator.process_turn()
   → route_message() → AGENT (tool_or_multistep_intent)

2. Supervisor graph:
   → supervisor routes to `parallel_rag` (comparison intent)
   → parallel_planner normalizes sub-tasks
   → rag_worker x N executes `run_rag_agent()` in parallel doc scopes

3. RAGAgent tool calls per worker (automated, no user input):
   a. resolve_document("termset_v1") → {"doc_id": "kb_abc123", "title": "termset_v1.pdf"}
   b. resolve_document("termset_v2") → {"doc_id": "kb_def456", "title": "termset_v2.pdf"}
   c. diff_documents("kb_abc123", "kb_def456")
      → {shared: [10,11,12,13,14,15], only_in_1: [], only_in_2: [15A]}
   d. scratchpad_write("diff", <diff result>)
   e. compare_clauses("kb_abc123", "kb_def456", ["10","11","12","13","14","15"])
      → side-by-side text for each clause

4. rag_synthesizer merges worker outputs
   → consolidated answer + citation list

5. supervisor routes to `__end__`
   → final answer returned to user

6. session.clear_scratchpad() (if CLEAR_SCRATCHPAD_PER_TURN=true)
```

---

## 14. Example Queries

### General Knowledge
```
You> What is the difference between HNSW and IVFFlat vector indexes?
```

### Document Search
```
You> What does our internal policy say about data retention?
```
```
You> Find the definition of "Force Majeure" in the uploaded contract.
```

### Clause Extraction
```
You> What does clause 33 say in the supply chain agreement?
```
```
You> Show me the full text of sections 10.1 through 10.4 in the termset.
```

### Requirements Extraction
```
You> Find all requirements from the specification document.
```
```
You> List all SHALL statements in doc.pdf that relate to data security.
```

### Document Comparison
```
You> What are the differences between termset_v1.pdf and termset_v2.pdf?
```
```
You> Go through both contracts and compare their clauses one by one.
```
```
You> Which clauses appear in contract_a but not in contract_b?
```

### Sequential Document Processing
```
You> First read the questions document, then look through the policy document
     and answer each question with citations.
```

### Upload and Query
```bash
python run.py ask -q "Summarise the key obligations" -u ./nda.pdf
```

```
You> /upload ./new_spec.pdf
You> Find all requirements related to delivery timelines from the spec I just uploaded.
```

### Memory
```
You> Remember that the contract value is £2.3 million.
You> (next session) What was the contract value we discussed?
```

### Math
```
You> What is 15% of £2,340,000?
```

### List Available Documents
```
You> What documents do you have access to?
```

---

## 15. Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| **Embedding model switch requires reindex** | Changing embed model dimensions requires rebuilding indexed vectors | Set `EMBEDDING_DIM` and run `python run.py migrate-embedding-dim --yes` |
| **No streaming output** | Full response is generated before display. Both agents use `graph.invoke()` (blocking). Streaming is architecturally possible now that both agents use LangGraph — replace with `graph.astream(stream_mode="messages")` | Use smaller models or increase `OLLAMA_NUM_PREDICT` until streaming is wired |
| **Single-process, synchronous** | One request at a time; no async concurrency | Wrap in a web server (FastAPI + asyncio) for concurrent users |
| **OCR quality depends on scan quality** | Very small text, rotated scans, or low-resolution images produce poor OCR | Increase `dpi` in `rag/ocr.py:_render_and_ocr_page` (default 200, try 300) |
| **PaddleOCR first-run download** | ~200 MB model download on first OCR use | Pre-pull in your Docker image or CI environment |
| **DOCX images not extracted** | Images embedded inside `.docx` files are ignored | Export to PDF first |
| **No multi-user isolation** | `memory` table is keyed by session UUID only; no authentication | Add an auth layer and pass user ID as session key |
| **Embedding dimension mismatch** | Provider output dims and DB vector column can drift | Run `python run.py doctor`, then `python run.py migrate-embedding-dim --yes` if misaligned |
| **Langfuse is a separate service** | Traces are silently dropped if Langfuse is unreachable | Set `LANGFUSE_PUBLIC_KEY=` (empty) to disable entirely |
| **No prompt-injection defences** | Agent processes document content with no sanitisation | Add a content-safety layer before ingestion in production |
| **HNSW index tuning** | Default `m=16, ef_construction=64` may not be optimal for very large collections (>1M chunks) | Tune in `schema.sql` and rebuild index |

---

## 16. Troubleshooting

### `connection refused` on PostgreSQL

```
psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1),
port 5432 failed: Connection refused
```

- Check the container is running: `docker ps | grep ragdb`
- If you use Docker Compose, check the DB service: `docker compose ps rag-postgres`
- Start it if stopped (Compose): `docker compose up -d rag-postgres`
- If you run a standalone container named `ragdb`, start it with: `docker start ragdb`
- Verify the `PG_DSN` in your `.env` matches the container credentials

---

### `vector type does not exist`

```
UndefinedObject: type "vector" does not exist
```

The `pgvector` extension is not loaded. Verify you are using the `pgvector/pgvector:pg16` Docker image (not plain `postgres:16`), then run:

```bash
python run.py migrate
```

---

### `ModuleNotFoundError: No module named 'langchain_ollama'`

This means your current Python environment is missing provider packages required by your selected provider settings (for example `LLM_PROVIDER=ollama`).

Run:

```bash
python run.py doctor
```

Host (venv) fix:

```bash
python -m pip install -r requirements.txt
python -m pip install langchain-ollama
```

Docker fix:

```bash
docker compose up -d --build app
docker compose restart app
docker compose exec app python run.py doctor
```

---

### Ollama model not found

```
OllamaError: model 'qwen3:8b' not found
```

Pull the model:

```bash
docker compose exec ollama ollama pull qwen3:8b
```

Or set `OLLAMA_CHAT_MODEL` in `.env` to a model you have pulled.

---

### PaddleOCR not installed warning

```
WARNING: PaddleOCR is not installed. Image files and scanned PDFs will be skipped.
```

Install the packages:

```bash
pip install paddlepaddle paddleocr pymupdf pillow
```

Or disable OCR entirely: `USE_PADDLE_OCR=false` in `.env`.

---

### Embedding dimension mismatch

```
ValueError: expected vector of dimension 768, got 1536
```

You changed embedding models without updating `EMBEDDING_DIM` or the schema. Fix:

1. Set the correct `EMBEDDING_DIM` in `.env` (for ada-002, use `1536`)
2. Run `python run.py doctor` (verify mismatch is detected)
3. Run `python run.py migrate-embedding-dim --yes`
4. Re-run your demo or `python run.py init-kb`

---

### OCR returns empty results

- Check the image resolution (blur, low DPI). Try increasing DPI in `rag/ocr.py` → `_render_and_ocr_page(dpi=300)`.
- Verify the language code: `OCR_LANGUAGE=en` for English, `ch` for Chinese, etc.
- Check PaddleOCR logs by temporarily setting `show_log=True` in `rag/ocr.py:get_ocr_engine`.

---

### `No content extracted from <file> — skipping`

The loader returned no text (empty file, corrupted PDF, or unsupported format). Check:
- The file is not password-protected
- For scanned PDFs: ensure `USE_PADDLE_OCR=true` and PaddleOCR is installed

---

## 17. Environment Variable Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `DATABASE_BACKEND` | `postgres` | Yes | Database backend (currently: `postgres`) |
| `VECTOR_STORE_BACKEND` | `pgvector` | Yes | Vector backend (currently: `pgvector`) |
| `OBJECT_STORE_BACKEND` | `local` | No | Object/doc source backend (`local`, `s3`, `azure_blob`; local implemented) |
| `SKILLS_BACKEND` | `local` | No | Skills prompt backend (`local`, `s3`, `azure_blob`; local implemented) |
| `PROMPTS_BACKEND` | `local` | No | Prompt-template backend (`local`, `s3`, `azure_blob`; local implemented) |
| `LLM_PROVIDER` | `azure` | Yes | `ollama`, `azure`, or `nvidia` |
| `JUDGE_PROVIDER` | same as `LLM_PROVIDER` | No | Provider used for grading/judge LLM |
| `EMBEDDINGS_PROVIDER` | same as `LLM_PROVIDER` | No | `ollama` or `azure` (set explicitly when `LLM_PROVIDER=nvidia`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | If Ollama | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | `qwen3:8b` | If Ollama | Chat model name |
| `OLLAMA_JUDGE_MODEL` | same as `OLLAMA_CHAT_MODEL` | No | Judge model name for Ollama |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | If Ollama | Embedding model name |
| `OLLAMA_TEMPERATURE` | `0.2` | No | Generation temperature |
| `OLLAMA_NUM_PREDICT` | `2048` | No | Max output tokens |
| `DEMO_OLLAMA_NUM_PREDICT` | `2048` | No | Demo-only max output tokens override (`python run.py demo`) |
| `OLLAMA_GGUF_AUTO_IMPORT` | `false` | No | Enable one-shot GGUF model creation helper (requires `ollama-import` profile) |
| `OLLAMA_GGUF_MODEL_NAME` | — | If auto import | Model name to create with `ollama create` |
| `OLLAMA_GGUF_MODELFILE` | `/gguf/Modelfile` | If auto import | Modelfile path mounted inside Ollama/importer containers |
| `AZURE_OPENAI_API_KEY` | — | If Azure | Azure API key |
| `AZURE_OPENAI_ENDPOINT` | — | If Azure | Azure resource endpoint |
| `AZURE_OPENAI_API_VERSION` | `2024-05-01-preview` | If Azure | Azure API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | — | If Azure | Chat deployment name |
| `AZURE_OPENAI_JUDGE_DEPLOYMENT` | same as `AZURE_OPENAI_CHAT_DEPLOYMENT` | No | Judge deployment name |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | — | If Azure embed | Embedding deployment name |
| `AZURE_TEMPERATURE` | `0.2` | No | Azure generation temperature |
| `NVIDIA_OPENAI_ENDPOINT` | — | If NVIDIA chat/judge | OpenAI-compatible base URL for NVIDIA endpoint |
| `NVIDIA_API_TOKEN` | — | If NVIDIA chat/judge | Bearer token for NVIDIA endpoint auth (`Token` alias also accepted) |
| `NVIDIA_CHAT_MODEL` | — | If `LLM_PROVIDER=nvidia` | NVIDIA chat model ID |
| `NVIDIA_JUDGE_MODEL` | same as `NVIDIA_CHAT_MODEL` | If `JUDGE_PROVIDER=nvidia` | NVIDIA judge model ID |
| `NVIDIA_TEMPERATURE` | `0.0` | No | NVIDIA chat/judge temperature |
| `JUDGE_TEMPERATURE` | `0.0` | No | Judge-model temperature |
| `PG_DSN` | `postgresql://raguser:ragpass@localhost:5432/ragdb` | Yes | PostgreSQL connection string |
| `RAG_DB_NAME` | `ragdb` | No | Compose-managed primary DB name |
| `RAG_DB_USER` | `raguser` | No | Compose-managed primary DB user |
| `RAG_DB_PASSWORD` | `ragpass` | No | Compose-managed primary DB password |
| `RAG_DB_PORT` | `5432` | No | Host port for compose Postgres |
| `EMBEDDING_DIM` | `1536` | Yes | Must match embed model output |
| `MAX_AGENT_STEPS` | `10` | No | GeneralAgent max loop iterations |
| `MAX_TOOL_CALLS` | `12` | No | Max tool calls per turn |
| `MAX_RAG_AGENT_STEPS` | `8` | No | RAGAgent max tool calls |
| `RAG_TOPK_VECTOR` | `12` | No | Chunks returned from vector search |
| `RAG_TOPK_BM25` | `12` | No | Chunks returned from keyword search |
| `RAG_MAX_RETRIES` | `2` | No | Accepted for compatibility (currently not wired into active runtime flow) |
| `RAG_MIN_EVIDENCE_CHUNKS` | `2` | No | Reserved config (currently not enforced in active runtime flow) |
| `SUPERVISOR_MAX_LOOPS` | `5` | No | Max supervisor routing loops per turn |
| `MAX_PARALLEL_RAG_WORKERS` | `4` | No | Max parallel RAG workers for document comparison |
| `ENABLE_PARALLEL_RAG` | `true` | No | Enable parallel RAG via Send API |
| `CHUNK_SIZE` | `900` | No | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | No | Character overlap between chunks |
| `DATA_DIR` | `./data` | No | Base data directory |
| `KB_DIR` | `./data/kb` | No | Knowledge base directory |
| `UPLOADS_DIR` | `./data/uploads` | No | User uploads directory |
| `KB_SOURCE_URI` | `file://./data/kb` | No | Knowledge base source URI (future remote support) |
| `UPLOADS_SOURCE_URI` | `file://./data/uploads` | No | Upload source URI (future remote support) |
| `SKILLS_DIR` | `./data/skills` | No | Skills directory root |
| `PROMPTS_DIR` | `./data/prompts` | No | Prompt-template directory root |
| `SHARED_SKILLS_PATH` | `./data/skills/skills.md` | No | Shared skills markdown path |
| `GENERAL_AGENT_SKILLS_PATH` | `./data/skills/general_agent.md` | No | General-agent skills path |
| `RAG_AGENT_SKILLS_PATH` | `./data/skills/rag_agent.md` | No | RAG-agent skills path |
| `SUPERVISOR_AGENT_SKILLS_PATH` | `./data/skills/supervisor_agent.md` | No | Supervisor-agent skills path |
| `UTILITY_AGENT_SKILLS_PATH` | `./data/skills/utility_agent.md` | No | Utility-agent skills path |
| `BASIC_CHAT_SKILLS_PATH` | `./data/skills/basic_chat.md` | No | Basic-chat skills path |
| `JUDGE_GRADING_PROMPT_PATH` | `./data/prompts/judge_grading.txt` | No | Grading prompt template path |
| `JUDGE_REWRITE_PROMPT_PATH` | `./data/prompts/judge_rewrite.txt` | No | Query-rewrite prompt template path |
| `GROUNDED_ANSWER_PROMPT_PATH` | `./data/prompts/grounded_answer.txt` | No | Grounded-answer prompt template path |
| `RAG_SYNTHESIS_PROMPT_PATH` | `./data/prompts/rag_synthesis.txt` | No | RAG synthesis prompt template path |
| `PARALLEL_RAG_SYNTHESIS_PROMPT_PATH` | `./data/prompts/parallel_rag_synthesis.txt` | No | Parallel-RAG synthesis prompt template path |
| `WAIT_FOR_DB` | `true` | No | App container waits for DB before startup commands |
| `DB_WAIT_TIMEOUT_SECONDS` | `90` | No | DB wait timeout in app entrypoint |
| `APP_AUTO_MIGRATE` | `true` | No | Auto-run `python run.py migrate` in app container entrypoint |
| `APP_AUTO_INIT_KB` | `false` | No | Auto-run `python run.py init-kb` in app container entrypoint |
| `APP_API_PORT` | `8000` | No | Host port mapped to gateway service in Docker Compose |
| `OLLAMA_BASE_URL_DOCKER` | `http://ollama:11434` | No | Internal Ollama URL injected into app service in compose |
| `LANGFUSE_HOST_DOCKER` | `http://langfuse-web:3000` | No | Internal Langfuse URL injected into app service in compose |
| `DEFAULT_TENANT_ID` | `local-dev` | No | Default tenant for CLI/demo/local runs |
| `DEFAULT_USER_ID` | `local-cli` | No | Default user for CLI/demo/local runs |
| `DEFAULT_CONVERSATION_ID` | `local-session` | No | Default conversation scope for CLI/demo/local runs |
| `GATEWAY_MODEL_ID` | `enterprise-agent` | No | Public model ID exposed by `/v1/models` |
| `CLEAR_SCRATCHPAD_PER_TURN` | `true` | No | Wipe scratchpad after each turn |
| `USE_PADDLE_OCR` | `true` | No | Enable PaddleOCR for images and scanned PDFs |
| `OCR_LANGUAGE` | `en` | No | PaddleOCR language code |
| `OCR_USE_GPU` | `false` | No | Use GPU for OCR (requires CUDA) |
| `OCR_MIN_PAGE_CHARS` | `50` | No | PDF pages with fewer chars trigger OCR |
| `LANGFUSE_HOST` | `http://localhost:3000` | No | Langfuse server URL |
| `LANGFUSE_PUBLIC_KEY` | — | No | Langfuse public key (leave empty to disable) |
| `LANGFUSE_SECRET_KEY` | — | No | Langfuse secret key |
| `LANGFUSE_DEBUG` | `false` | No | Enable Langfuse debug logging |
| `LANGFUSE_WEB_PORT` | `3000` | No | Host port for Langfuse UI in compose |
| `LANGFUSE_POSTGRES_PASSWORD` | `postgres` | No | Langfuse Postgres password in compose |
| `LANGFUSE_REDIS_AUTH` | `myredissecret` | No | Langfuse Redis password in compose |
| `LANGFUSE_CLICKHOUSE_PASSWORD` | `clickhouse` | No | Langfuse ClickHouse password in compose |
| `LANGFUSE_MINIO_ROOT_USER` | `minio` | No | Langfuse MinIO user in compose |
| `LANGFUSE_MINIO_ROOT_PASSWORD` | `miniosecret` | No | Langfuse MinIO password in compose |
| `LANGFUSE_NEXTAUTH_SECRET` | `mysecret` | No | Langfuse web auth secret in compose |
| `LANGFUSE_SALT` | `mysalt` | No | Langfuse salt value in compose |
| `LANGFUSE_ENCRYPTION_KEY` | `000...000` | No | 32-char Langfuse encryption key |

---

## 18. Project Layout

```
langchain_agentic_chatbot_v2/
│
├── run.py                         # Entry point — adds src/ to PYTHONPATH
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for environment variables
├── Dockerfile                     # App container image
├── docker-compose.yml             # Full local stack: app + db + optional ollama/langfuse
├── docker/
│   └── entrypoint.sh              # App startup (wait for DB + optional auto migrate/init)
│
├── data/
│   ├── kb/                        # Built-in knowledge base documents
│   │   ├── 01_product_overview.md
│   │   ├── 02_pricing_and_plans.md
│   │   ├── 03_security_and_privacy.md
│   │   ├── 04_integrations_and_tools.md
│   │   ├── 05_release_notes.md
│   │   ├── 06_master_services_agreement_v1.md
│   │   ├── 07_master_services_agreement_v2.md
│   │   ├── 08_data_processing_addendum_global.md
│   │   ├── 09_ai_ops_control_standard.md
│   │   ├── 10_incident_communications_playbook.md
│   │   ├── 11_vendor_security_schedule.md
│   │   ├── api_auth.md
│   │   ├── api_endpoints.md
│   │   ├── api_examples.md
│   │   ├── api_rate_limits.md
│   │   ├── runbook_data_pipeline.md
│   │   ├── runbook_incident_response.md
│   │   └── runbook_oncall_handover.md
│   ├── ollama/
│   │   └── gguf/                 # Optional GGUF drop folder + Modelfile
│   ├── skills/                    # Agent system prompts (edit without code changes)
│   │   ├── skills.md              # Shared context injected into all agents
│   │   ├── general_agent.md       # GeneralAgent + BasicChat instructions + few-shot examples
│   │   ├── rag_agent.md           # RAGAgent decision trees + failure recovery guide
│   │   ├── supervisor_agent.md    # Supervisor routing rules (multi-agent graph)
│   │   ├── utility_agent.md       # Utility agent instructions
│   │   └── basic_chat.md          # (optional) Dedicated BasicChat prompt
│   ├── prompts/                   # Judge/synthesis prompt templates (path-configurable)
│   │   ├── judge_grading.txt
│   │   ├── judge_rewrite.txt
│   │   ├── grounded_answer.txt
│   │   ├── rag_synthesis.txt
│   │   └── parallel_rag_synthesis.txt
│   ├── demo/                      # Curated demo scenario definitions
│   │   └── demo_scenarios.json
│   └── uploads/                   # Runtime upload directory
│
├── docs/                          # Additional architecture and design docs
│   ├── ARCHITECTURE.md            # System architecture, LangGraph flows, design decisions
│   ├── C4_ARCHITECTURE.md         # C4 context/container/component diagrams (code-accurate)
│   ├── PATTERNS.md                # Agentic patterns implemented + summary table
│   ├── RAG_AGENT_DESIGN.md        # RAGAgent loop design, all 11 tools, output contract
│   ├── TOOLS_AND_TOOL_CALLING.md  # Tool design principles, LangGraph loop, all tool schemas
│   ├── RAG_TOOL_CONTRACT.md       # Full rag_agent_tool output schema specification
│   ├── PROVIDERS.md               # Ollama vs Azure OpenAI provider configuration
│   ├── OBSERVABILITY_LANGFUSE.md  # Langfuse tracing setup and trace structure
│   ├── OPENAI_GATEWAY.md           # OpenAI-compatible /v1 gateway + OpenWebUI/AI SDK wiring
│   ├── ROUTER_RUBRIC.md           # Router decision rules and confidence scoring
│   ├── KB_DEMO_PACKS.md           # Built-in knowledge base document descriptions
│   ├── SKILLS_PLAYBOOK.md         # Skill authoring and governance playbook
│   └── COMPOSITION.md             # How the system components compose together
│
└── src/agentic_chatbot/
    ├── api/
    │   ├── __init__.py            # FastAPI gateway package export
    │   └── main.py                # OpenAI-compatible /v1 endpoints (no in-app auth)
    ├── cli.py                     # Typer CLI (ask, chat, demo, migrate, init-kb, reset-indexes, serve-api)
    ├── config.py                  # Settings dataclass + load_settings() from env
    ├── context.py                 # RequestContext + local default context resolver
    ├── demo/                      # Structured scenario parser + verification helpers
    ├── prompting.py               # Prompt template loading + token replacement
    │
    ├── agents/
    │   ├── orchestrator.py        # ChatbotApp — top-level router + multi-agent graph coordinator
    │   ├── general_agent.py       # Tool-calling loop agent (fallback path)
    │   ├── basic_chat.py          # Direct LLM call (no tools)
    │   └── session.py             # ChatSession (messages, scratchpad, uploaded_doc_ids)
    │
    ├── graph/                     # Multi-agent supervisor graph (LangGraph)
    │   ├── builder.py             # Assembles and compiles the StateGraph
    │   ├── state.py               # AgentState TypedDict with message reducers
    │   ├── supervisor.py          # Supervisor LLM node + response parser
    │   ├── session_proxy.py       # Duck-typing bridge for tool compatibility
    │   └── nodes/
    │       ├── rag_node.py        # Single RAG agent node (wraps run_rag_agent)
    │       ├── utility_node.py    # Utility agent subgraph (calc, memory, list_docs)
    │       ├── parallel_planner_node.py  # Validates sub-tasks before Send fan-out
    │       ├── rag_worker_node.py        # Parallel RAG worker (one per Send)
    │       └── rag_synthesizer_node.py   # Merges parallel RAG results
    │
    ├── db/
    │   ├── schema.sql             # PostgreSQL DDL (tables, indexes, extensions)
    │   ├── connection.py          # ThreadedConnectionPool singleton
    │   ├── chunk_store.py         # Vector search, keyword search, clause extraction
    │   ├── document_store.py      # Document metadata CRUD + fuzzy title search
    │   ├── memory_store.py        # Cross-turn key-value memory
    │   └── migration.py           # One-time migration helper from legacy Chroma/BM25
    │
    ├── providers/
    │   └── llm_factory.py         # ProviderBundle factory (Ollama / Azure OpenAI)
    │
    ├── rag/
    │   ├── agent.py               # Loop-based RAGAgent with 11 specialist tools
    │   ├── ingest.py              # File loading → structure detection → chunking → DB
    │   ├── ocr.py                 # PaddleOCR singleton + image/PDF OCR functions
    │   ├── structure_detector.py  # Regex-based doc classifier (no LLM)
    │   ├── clause_splitter.py     # Clause-boundary document splitter
    │   ├── retrieval.py           # vector_search / keyword_search / hybrid wrappers
    │   ├── skills.py              # Runtime loader for data/skills/*.md prompt files
    │   ├── stores.py              # KnowledgeStores dataclass (ChunkStore, DocStore, MemStore)
    │   ├── answer.py              # Grounded answer synthesis
    │   ├── grading.py             # LLM-based chunk relevance grading
    │   └── rewrite.py             # Query rewriting for failed retrievals
    │
    ├── router/
    │   └── router.py              # Deterministic regex router (BASIC vs AGENT)
    │
    ├── tools/
    │   ├── calculator.py          # Simple math tool
    │   ├── list_docs.py           # list_indexed_docs tool
    │   ├── memory_tools.py        # memory_save / memory_load / memory_list tools
    │   ├── rag_agent_tool.py      # rag_agent_tool — wraps RAGAgent as a LangChain tool
    │   └── rag_tools.py           # 11 RAG specialist tools (resolve, search, extract, compare, ...)
    │
    └── observability/
        └── callbacks.py           # Langfuse LangChain callback wiring
```

---

## Notes on Production Readiness

This project is a reference implementation and teaching tool. Before deploying in production, consider adding:

- **Input sanitisation** — prevent prompt injection via document content
- **Authentication and authorisation** — isolate users by adding user IDs to session keys
- **Rate limiting** — protect against runaway agent loops
- **Streaming output** — both agents use LangGraph; streaming requires replacing `graph.invoke()` with `graph.astream(stream_mode="messages")` in `general_agent.py` and `rag/agent.py`
- **Session persistence (checkpointing)** — add `PostgresSaver` from `langgraph-checkpoint-postgres` as the `checkpointer` argument to `create_react_agent()`. Conversations will survive restarts, and time-travel debugging becomes available
- **Structured output grading** — replace fragile `extract_json()` parsing with `llm.with_structured_output(RagAnswer)` Pydantic models in `rag/agent.py` and `rag/grading.py`
- **Parallel retrieval** — the multi-agent graph already supports parallel RAG via the Send API; tune `MAX_PARALLEL_RAG_WORKERS` for your workload
- **Error boundaries** — graceful degradation when LLM or DB is unavailable
- **CI/CD** — automated tests and schema migration pipeline
- **Monitoring** — track token usage, latency, and retrieval quality via Langfuse or another observability tool
