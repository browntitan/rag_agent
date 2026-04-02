# `new_demo_notebook`

This folder is an API-driven Jupyter showcase for the live `agentic_chatbot_next` backend runtime.

It does not run a separate demo runtime. The notebook starts the real FastAPI
gateway from this repository, calls `/v1/chat/completions` and
`/v1/ingest/documents`, then renders orchestration traces from `data/runtime`.
The notebook is self-checking: it asserts preflight readiness, scenario success,
and trace persistence while it runs.

## What it demonstrates

- BASIC vs AGENT routing
- General-agent tool use
- Grounded RAG through the public API
- Data-analyst CSV workflows using the session workspace
- Coordinator orchestration with planner, workers, finalizer, and verifier
- Background `memory_maintainer` jobs and task notifications

## Setup

1. Install the main repository requirements.
2. Install notebook-side extras:

```bash
python -m pip install -r new_demo_notebook/requirements.txt
```

3. Make sure the backend provider and database environment is configured.
4. Launch Jupyter:

```bash
jupyter notebook new_demo_notebook/agentic_system_showcase.ipynb
```

The notebook starts and stops the live backend server itself with `python run.py serve-api`
and drives the real `/v1/chat/completions` and `/v1/ingest/documents` routes from this
repository.

For the verified long-timeout Ollama acceptance profile, export:

```bash
export PYTHONPATH=src
export LLM_PROVIDER=ollama
export EMBEDDINGS_PROVIDER=ollama
export JUDGE_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_CHAT_MODEL=qwen3.5:9b
export OLLAMA_JUDGE_MODEL=qwen3.5:9b
export OLLAMA_EMBED_MODEL=nomic-embed-text:latest
export EMBEDDING_DIM=768
export DEFAULT_COLLECTION_ID=default
export SANDBOX_DOCKER_IMAGE=agentic-chatbot-sandbox:py312

export NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS=900
export NEXT_RUNTIME_JOB_WAIT_SECONDS=300
export NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS=300
export NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS=5400
export SANDBOX_TIMEOUT_SECONDS=300
```

`NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS` controls how long the notebook helper waits for
the live API server to report `/health/ready`. `NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS`
controls the `nbconvert` execution timeout used by the notebook acceptance harness.

Run the local acceptance flow in this order:

```bash
python -m pip install -r new_demo_notebook/requirements.txt
ollama list
docker info
python run.py doctor --strict
python run.py migrate
python run.py sync-kb --collection-id default
python run.py index-skills
RUN_NEXT_RUNTIME_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py
RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py -k notebook
```

To run the same scenario manifest as a manual acceptance harness outside Jupyter:

```bash
RUN_NEXT_RUNTIME_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py
```

To execute the actual notebook as a gated smoke test and write the executed copy
to `new_demo_notebook/.artifacts/executed/`:

```bash
RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py -k notebook
```

## Notes

- Server logs are written to `new_demo_notebook/.artifacts/server.log`.
- Executed notebook output is written to
  `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`.
- Runtime traces are read from `data/runtime`.
- Session workspaces live under `data/workspaces/`.
- Memory artifacts live under `data/memory/`.
- Scenario definitions live in `new_demo_notebook/scenarios/scenarios.json`.
