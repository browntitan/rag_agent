# Local Docker Stack

This runbook documents the local Docker-backed stack in the `-codex-next-task` worktree.

The important defaults in this worktree are now:

- `LLM_PROVIDER=ollama`
- `JUDGE_PROVIDER=ollama`
- `EMBEDDINGS_PROVIDER=ollama`
- `OLLAMA_CHAT_MODEL=gpt-oss:20b`
- `OLLAMA_JUDGE_MODEL=gpt-oss:20b`
- `OLLAMA_EMBED_MODEL=nomic-embed-text:latest`
- `KB_EXTRA_DIRS=./docs`

That last setting matters for grounded architecture questions because it indexes `docs/*.md`
alongside `data/kb/*`.

## Prerequisites

- Docker Desktop or Docker Engine with Compose
- Python 3.12+
- Node.js 20+
- enough free disk for `gpt-oss:20b`

## 0. Reset to a clean local state

Stop any running local `serve-api` or Vite processes first.

Reset containers while keeping downloaded Ollama models:

```bash
docker compose down --remove-orphans
rm -rf data/runtime data/memory data/uploads data/workspaces
mkdir -p data/runtime data/memory data/uploads data/workspaces
```

If you want a full Docker reset, including Compose-managed volumes:

```bash
docker compose down --remove-orphans -v
```

## 1. Create the local env file

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2-codex-next-task
cp .env.example .env
```

The default template in this worktree is already Ollama-first and points KB sync at `./docs`.

## 2. Start the backend dependencies

```bash
docker compose up -d rag-postgres
docker compose --profile ollama up -d ollama
```

If you want Langfuse too:

```bash
docker compose --profile observability up -d
```

## 3. Pull the models used by this worktree

If Ollama is running on the host:

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text:latest
```

If you are using the Dockerized Ollama service:

```bash
docker compose exec ollama ollama pull gpt-oss:20b
docker compose exec ollama ollama pull nomic-embed-text:latest
```

The first `gpt-oss:20b` pull can take a while.

## 4. Start the backend

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py doctor --strict
python run.py migrate
python run.py index-skills
python run.py sync-kb --collection-id default
python run.py serve-api --host 0.0.0.0 --port 8000
```

`doctor --strict` now reports KB corpus coverage. `serve-api` also auto-syncs configured KB/docs
sources on startup by default. If startup sync is disabled or fails, `/health/ready` returns `503`
with `reason`, `missing_sources`, and the suggested `sync-kb` command.

For the common local dev path, treat `doctor --strict` as the gate. If it passes, start
`serve-api` and move on to the frontend. Only fall back to `migrate`, `index-skills`, and
`sync-kb` when doctor reports schema, provider, or KB coverage problems.

## 5. Start the frontend

In another terminal:

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2-codex-next-task/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

- app UI: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Langfuse UI: [http://127.0.0.1:3000](http://127.0.0.1:3000) if enabled

Upload endpoint note:

- the frontend uses `POST /v1/upload` for multipart file uploads
- `POST /v1/ingest/documents` is the path-based ingest endpoint and the guaranteed pre-chat
  workspace-seeding path
- treat `/v1/upload` as the frontend KB-ingest path, not the canonical way to pre-seed the
  analyst workspace
- the current `/v1/upload` workspace copy is a best-effort legacy convenience and does not
  open the canonical `filesystem_key(session_id)` workspace path for you

If Vite logs `http proxy error` for `/v1/...` and the UI shows a blank `HTTP 500:` style error,
the frontend is up but the backend is not listening on port `8000`. Start it with
`python run.py serve-api --host 0.0.0.0 --port 8000`, then retry the prompt.

## 6. Verify that RAG has the repo docs

Useful checks:

```bash
python run.py doctor --strict
curl http://127.0.0.1:8000/health/ready
python run.py sync-kb --collection-id default
```

You should see coverage for:

- `data/kb/*`
- `docs/ARCHITECTURE.md`
- `docs/C4_ARCHITECTURE.md`
- `docs/NEXT_RUNTIME_FOUNDATION.md`

## 7. Good smoke tests

- `Hello there`
- `What are the key implementation details in the architecture docs? Cite your sources.`
- `Compare the indexed services agreement and cite the differences.`

Expected architecture-doc behavior:

- citations from repo docs such as `ARCHITECTURE.md`, `C4_ARCHITECTURE.md`, and `NEXT_RUNTIME_FOUNDATION.md`
- no `No evidence available in the context` warning after KB sync
- if KB coverage is missing, the answer tells you to run `python run.py sync-kb --collection-id default`

For upload analysis, upload in the same conversation you plan to analyze. Fresh conversations
start with an empty workspace.
