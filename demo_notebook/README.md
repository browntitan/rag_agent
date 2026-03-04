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

## Quickstart

1. Create a notebook env and install dependencies:

```bash
cd demo_notebook
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

2. Configure settings:

```bash
cp .env.example .env
```

Optional skills showcase toggles in `.env`:

```env
NOTEBOOK_SKILLS_ENABLED=true
NOTEBOOK_SKILLS_DIR=./skills
NOTEBOOK_SKILLS_SHOWCASE_MODE=false
```

3. Run isolation check:

```bash
python scripts/check_isolation.py
```

4. Launch Jupyter:

```bash
jupyter notebook agentic_rag_showcase.ipynb
```

## Runtime prerequisites

- PostgreSQL with pgvector reachable via `NOTEBOOK_PG_DSN`.
- KB files available under `NOTEBOOK_KB_DIR` (default `../data/kb`).
- Provider endpoint configured for chosen `NOTEBOOK_PROVIDER`.

## Provider modes

### Azure
Set `NOTEBOOK_PROVIDER=azure` and configure:
- `NOTEBOOK_AZURE_API_KEY`
- `NOTEBOOK_AZURE_ENDPOINT` (Gov endpoints like `*.openai.azure.us` are valid)
- `NOTEBOOK_AZURE_CHAT_DEPLOYMENT`
- `NOTEBOOK_AZURE_JUDGE_DEPLOYMENT`
- `NOTEBOOK_AZURE_EMBED_DEPLOYMENT`

### Ollama
Set `NOTEBOOK_PROVIDER=ollama` and configure:
- `NOTEBOOK_OLLAMA_BASE_URL`
- `NOTEBOOK_OLLAMA_CHAT_MODEL`
- `NOTEBOOK_OLLAMA_JUDGE_MODEL`
- `NOTEBOOK_OLLAMA_EMBED_MODEL`
- `NOTEBOOK_EMBEDDING_DIM` (set `768` for `nomic-embed-text`)

### vLLM
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
