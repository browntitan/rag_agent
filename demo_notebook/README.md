# demo_notebook (Standalone Branch-Only Deliverable)

This folder is a fully standalone mini-product for demonstrating LangGraph agentic RAG in Jupyter.
It is intentionally isolated from the production app runtime.

## Documentation

- Technical guide: [docs/README_TECHNICAL.md](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/docs/README_TECHNICAL.md)
- C4 diagrams: [docs/C4_ARCHITECTURE.md](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/docs/C4_ARCHITECTURE.md)
- Skills usage: [docs/SKILLS_GUIDE.md](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/docs/SKILLS_GUIDE.md)
- Print trace observability: [docs/OBSERVABILITY_PRINT_TRACES.md](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/docs/OBSERVABILITY_PRINT_TRACES.md)

## Isolation guarantees

- No imports from `src/agentic_chatbot`.
- All implementation lives under `demo_notebook/runtime`.
- Existing KB files under `../data/kb` are reused as input corpus.
- Removing this folder removes the deliverable.

## Quickstart (New Machine)

1. Create a notebook env and install dependencies:

```bash
cd demo_notebook
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/check_isolation.py
```

2. Start required Docker services from repository root:

```bash
docker compose -f demo_notebook/docker-compose.yml up -d notebook-postgres
```

3. Launch notebook:

```bash
cd demo_notebook
source .venv/bin/activate
jupyter notebook agentic_rag_showcase.ipynb
```

Optional skills showcase toggles in `.env`:

```env
NOTEBOOK_SKILLS_ENABLED=true
NOTEBOOK_SKILLS_DIR=./skills
NOTEBOOK_SKILLS_SHOWCASE_MODE=false
```

## Docker for demo_notebook

Compose file: [docker-compose.yml](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/docker-compose.yml)

- `notebook-postgres` is required for all modes.
- `notebook-ollama` is optional and only needed when you run Ollama mode in a container.

Start Postgres only:

```bash
docker compose -f demo_notebook/docker-compose.yml up -d notebook-postgres
```

Start Postgres + Ollama:

```bash
docker compose -f demo_notebook/docker-compose.yml --profile ollama up -d notebook-postgres notebook-ollama
```

Check status:

```bash
docker compose -f demo_notebook/docker-compose.yml ps
```

Stop services:

```bash
docker compose -f demo_notebook/docker-compose.yml down
```

## Azure Setup

Use Docker only for Postgres. In `demo_notebook/.env` set:

```env
NOTEBOOK_PROVIDER=azure
NOTEBOOK_PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
NOTEBOOK_EMBEDDING_DIM=1536
NOTEBOOK_AZURE_API_KEY=...
NOTEBOOK_AZURE_ENDPOINT=https://YOUR-RESOURCE.openai.azure.us/
NOTEBOOK_AZURE_CHAT_DEPLOYMENT=gpt-4o
NOTEBOOK_AZURE_JUDGE_DEPLOYMENT=gpt-4o
NOTEBOOK_AZURE_EMBED_DEPLOYMENT=text-embedding-ada-002
```

## Ollama Setup

In `demo_notebook/.env` set:

```env
NOTEBOOK_PROVIDER=ollama
NOTEBOOK_PG_DSN=postgresql://raguser:ragpass@localhost:5432/ragdb
NOTEBOOK_OLLAMA_EMBED_MODEL=nomic-embed-text
NOTEBOOK_EMBEDDING_DIM=768
```

Use one Ollama path:

1. Host Ollama app:
- `NOTEBOOK_OLLAMA_BASE_URL=http://localhost:11434`
- Install models on host:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

2. Docker Ollama container:
- Start with compose profile:

```bash
docker compose -f demo_notebook/docker-compose.yml --profile ollama up -d notebook-ollama
```

- `NOTEBOOK_OLLAMA_BASE_URL=http://localhost:11434`

## Host Model Files -> Ollama Container (GGUF Import)

Use this when you download model files on host and want to import them into the container.

1. Put files in [data/ollama/gguf](/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/data/ollama/gguf):
- your `.gguf` file
- a `Modelfile`

Example host download commands:

```bash
# Option A: direct URL
curl -L -o demo_notebook/data/ollama/gguf/qwen3-8b-q4_k_m.gguf <GGUF_DOWNLOAD_URL>

# Option B: huggingface-cli
huggingface-cli download <repo-id> <file.gguf> \
  --local-dir demo_notebook/data/ollama/gguf
```

Example `Modelfile`:

```text
FROM /gguf/qwen3-8b-q4_k_m.gguf
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
```

2. Start container Ollama:

```bash
docker compose -f demo_notebook/docker-compose.yml --profile ollama up -d notebook-ollama
```

3. Create model inside container from mounted host files:

```bash
docker compose -f demo_notebook/docker-compose.yml exec notebook-ollama \
  ollama create qwen3-8b-local -f /gguf/Modelfile
```

4. Verify model exists:

```bash
docker compose -f demo_notebook/docker-compose.yml exec notebook-ollama ollama list
```

5. Point notebook to that model:

```env
NOTEBOOK_OLLAMA_CHAT_MODEL=qwen3-8b-local
NOTEBOOK_OLLAMA_JUDGE_MODEL=qwen3-8b-local
```

## vLLM Setup

Set `NOTEBOOK_PROVIDER=vllm` and configure:
- `NOTEBOOK_VLLM_BASE_URL`
- `NOTEBOOK_VLLM_CHAT_MODEL`
- optional `NOTEBOOK_VLLM_JUDGE_MODEL`
- optional embeddings endpoint model `NOTEBOOK_VLLM_EMBED_MODEL`

If your vLLM deployment does not expose embeddings, keep:

```env
NOTEBOOK_VLLM_USE_OPENAI_EMBEDDINGS=false
```

This enables local deterministic fallback embeddings for demo-only use.

## Demo notebook sections

- Setup and bootstrap (provider, DB schema, KB indexing)
- Scenario A: BASIC route
- Scenario B: AGENT RAG with citations
- Scenario C: parallel multi-doc orchestration trace
- Scenario D: explicit GeneralAgent path (no memory tools)
- Scenario E: provider switch notes
- Scenario F: skills showcase (baseline vs skills-enabled prompts)

## Skills showcase quick run

Section F creates two orchestrators for the same task:

1. Baseline prompt behavior (`skills_showcase_mode=false`)
2. Skills-composed behavior (`skills_showcase_mode=true`)

It prints which files from `demo_notebook/skills/` were applied so the prompt delta is explicit.

## Observability (no extra services)

The notebook uses print-based observability:
- LLM start/end
- tool start/end
- graph node update stream (supervisor routing, worker fan-out/fan-in)

No Langfuse containers are required.

## Remove this deliverable

Delete the folder:

```bash
rm -rf demo_notebook
```

No production app code depends on this folder.
