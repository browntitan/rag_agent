# Providers and Backend Config

## Provider roles

The runtime currently has three provider roles:

- chat LLM
- judge LLM
- embeddings

Supported today:

- chat LLM: `ollama`, `azure`, or `nvidia`
- judge LLM: `ollama`, `azure`, or `nvidia`
- embeddings: `ollama` or `azure`

Only the providers selected by `LLM_PROVIDER`, `JUDGE_PROVIDER`, and
`EMBEDDINGS_PROVIDER` are validated.

## What each provider role does

### Chat LLM

Used by:

- `run_basic_chat()`
- all `react`-mode runtime agents via `run_general_agent()`:
  - `general`
  - `utility`
  - `data_analyst`
  - `memory_maintainer`
- `run_rag_agent()` for the `rag_worker` path and upload-summary kickoff
- planner worker model calls
- finalizer worker model calls
- verifier worker model calls

### Judge LLM

Used by:

- LLM-router escalation inside `route_turn()` when deterministic routing confidence is low
- `run_rag_agent()` grading / grounded-answer support

### Embeddings

Used by:

- KB ingest and retrieval
- skill-pack indexing and retrieval

## Ollama example

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

## Azure OpenAI example

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
EMBEDDING_DIM=1536
```

## Mixed-provider setups

Mixed setups are allowed. Example:

```bash
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=ollama
```

## Operational checks

Useful commands:

```bash
python run.py doctor
python run.py migrate
python run.py index-skills
python run.py sync-kb
```

## Related runtime settings

The provider layer now feeds the live next runtime directly. Important nearby settings are:

- `LLM_ROUTER_ENABLED`
- `LLM_ROUTER_CONFIDENCE_THRESHOLD`
- `ENABLE_COORDINATOR_MODE`
- `RUNTIME_EVENTS_ENABLED`

Those settings do not change provider construction, but they do affect how the runtime uses
the configured models.
