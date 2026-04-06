# Next Runtime Status

`src/agentic_chatbot_next/` is no longer just a foundation package. It is the live runtime
used by the CLI and FastAPI gateway.

This is also an intentional breaking change for in-process callers: the removed
`agentic_chatbot.runtime.*` package and `agentic_chatbot.agents.orchestrator.ChatbotApp`
compatibility layer are no longer supported import paths.

## What changed

The original purpose of `agentic_chatbot_next` was to stage a cleaner runtime with:

- filesystem-safe runtime paths
- markdown-frontmatter agent definitions
- typed runtime contracts
- file-backed memory
- a cleaner session-kernel boundary

That cutover is now complete enough that the live entrypoints use the next runtime.

## Current role in the repository

Live runtime surface:

- `src/agentic_chatbot_next/app/service.py`
- `src/agentic_chatbot_next/runtime/*`
- `src/agentic_chatbot_next/router/*`
- `src/agentic_chatbot_next/tools/*`
- `src/agentic_chatbot_next/memory/*`
- `data/agents/*.md`

## Live source of truth

### Agent definitions

The live runtime resolves agents from `data/agents/*.md`. Markdown frontmatter is now the
authoritative agent-definition format.

Historical JSON agent artifacts are not part of the current runtime contract.

### Runtime paths

Runtime artifacts are keyed through `filesystem_key(...)` in
`src/agentic_chatbot_next/runtime/context.py`.

That applies to:

- session directories
- job directories
- workspace directories
- memory directories

### Memory

Live memory is file-backed under `data/memory/...`.

Authoritative files:

- `index.json` per scope

Derived files:

- `MEMORY.md`
- `topics/*.md`

The PostgreSQL memory table is not the live memory path for `agentic_chatbot_next`.

## Cutover status

The hard cut is complete: the live runtime now owns its config, provider factories, Postgres
primitives, sandbox exceptions, and low-level ingest helpers under `src/agentic_chatbot_next/`.

The import-boundary test now enforces that runtime code, tests, examples, and notebook helpers
do not import `agentic_chatbot.*`.

## Acceptance verification

The authoritative live acceptance gate is the optional scenario harness plus the executed notebook
smoke in `tests/test_next_acceptance_harness.py`.

Verified local operator flow:

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

The long-timeout harness-only env vars for this flow are:

- `NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS`
- `NEXT_RUNTIME_JOB_WAIT_SECONDS`
- `NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS`
- `NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS`
- `SANDBOX_TIMEOUT_SECONDS`

`NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS` and
`NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS` are operator env knobs for the notebook and
acceptance helpers only. They are not part of `config.Settings`, the public CLI contract, or
the public API surface.

`new_demo_notebook/` remains a supported harness for demos and acceptance coverage, but it is
support infrastructure around the live next runtime rather than part of the runtime package
boundary itself.

Acceptance evidence is written to:

- `new_demo_notebook/.artifacts/server.log`
- `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`
- `data/runtime/sessions/<filesystem_key(session_id)>/`
- `data/runtime/jobs/<filesystem_key(job_id)>/`
- `data/workspaces/<filesystem_key(session_id)>/`
- `data/memory/tenants/<tenant>/users/<user>/...`
