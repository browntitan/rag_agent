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

The OpenAI-compatible gateway model ID is separate from the underlying runtime model names.
Keep `GATEWAY_MODEL_ID=enterprise-agent` stable unless you intentionally want to change the
public API contract.

## What each provider role does

### Chat LLM

Used by:

- `run_basic_chat()`
- all `react`-mode runtime agents via `run_general_agent()`:
  - `general`
  - `utility`
  - `data_analyst`
- `run_rag_contract()` for the `rag_worker` path and upload-summary kickoff
- planner worker model calls
- finalizer worker model calls
- verifier worker model calls

The dedicated `memory_maintainer` mode does not currently use chat or judge providers. It
runs local heuristic extraction.

### Judge LLM

Used by:

- LLM-router escalation inside `route_turn()` when deterministic routing confidence is low
- `run_rag_contract()` grading / grounded-answer support
- `rag_agent_tool` calls made by `general` or `verifier`

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
OLLAMA_CHAT_MODEL=gpt-oss:20b
OLLAMA_JUDGE_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text:latest
OLLAMA_TEMPERATURE=0.2
JUDGE_TEMPERATURE=0.0
EMBEDDING_DIM=768
KB_EXTRA_DIRS=./docs
```

## Per-agent runtime overrides

The live runtime can override chat and judge model selection per agent role without changing
the public gateway model ID.

Environment pattern:

```bash
AGENT_<AGENT_NAME>_CHAT_MODEL=...
AGENT_<AGENT_NAME>_JUDGE_MODEL=...
```

Examples:

```bash
AGENT_GENERAL_CHAT_MODEL=gpt-oss:20b
AGENT_DATA_ANALYST_CHAT_MODEL=gpt-oss:20b
AGENT_MEMORY_MAINTAINER_JUDGE_MODEL=gpt-oss:20b
```

Notes:

- agent names are normalized to lowercase with underscores
- if no per-agent override is set, the agent inherits the shared provider defaults
- overrides affect chat and judge only; embeddings remain shared/global
- overrides only matter for roles that actually invoke providers; today
  `memory_maintainer` overrides are effectively inert

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
