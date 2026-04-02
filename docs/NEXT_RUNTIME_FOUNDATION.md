# Next Runtime Status

`src/agentic_chatbot_next/` is no longer just a foundation package. It is the live runtime
used by the CLI and FastAPI gateway.

## What changed

The original purpose of `agentic_chatbot_next` was to stage a cleaner runtime with:

- filesystem-safe runtime paths
- markdown-frontmatter agent definitions
- typed runtime contracts
- file-backed memory
- a cleaner session-kernel boundary

That cutover is now complete enough that the live entrypoints use the next runtime.

## Current role in the repository

Live:

- `src/agentic_chatbot_next/app/service.py`
- `src/agentic_chatbot_next/runtime/*`
- `src/agentic_chatbot_next/router/*`
- `src/agentic_chatbot_next/tools/*`
- `src/agentic_chatbot_next/memory/*`
- `data/agents/*.md`

Compatibility/reference only:

- `src/agentic_chatbot/agents/orchestrator.py` as a deprecated shim
- `src/agentic_chatbot/runtime/*` as legacy reference code

## Live source of truth

### Agent definitions

The live runtime resolves agents from `data/agents/*.md`. Markdown frontmatter is now the
authoritative agent-definition format.

The legacy `data/agents/*.json` artifacts have been retired from the live repo state and are
no longer part of the runtime contract.

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

The legacy PostgreSQL memory table is not the live memory path for `agentic_chatbot_next`.

## What is still shared

`agentic_chatbot_next` is live, but not every low-level primitive has been rewritten from
scratch yet.

Shared infrastructure that may still come from `src/agentic_chatbot/`:

- provider factories
- DB primitives
- `config.Settings`
- sandbox exception types
- low-level OCR / clause-splitting / structure-detection helpers

This boundary is intentional and enforced by the next-runtime import-boundary test. Runtime-
facing orchestration must not import legacy router, runtime-kernel, or orchestrator modules.

What matters operationally is that runtime-facing orchestration now lives behind the next
runtime modules and the public API/CLI surface is driven by them.

## Acceptance verification

The authoritative live acceptance gate is the scenario harness plus the executed notebook
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

Acceptance evidence is written to:

- `new_demo_notebook/.artifacts/server.log`
- `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`
- `data/runtime/sessions/<filesystem_key(session_id)>/`
- `data/runtime/jobs/<filesystem_key(job_id)>/`
- `data/workspaces/<filesystem_key(session_id)>/`
- `data/memory/tenants/<tenant>/users/<user>/...`
