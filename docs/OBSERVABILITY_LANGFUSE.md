# Observability with Langfuse and Local Runtime Events

This project has two observability layers:

1. LangChain callbacks, usually backed by Langfuse
2. durable local runtime artifacts written by the live `agentic_chatbot_next` runtime

Langfuse is optional. The authoritative local ground truth for session and worker behavior is
the file-backed persistence written from `src/agentic_chatbot_next/runtime/*`.

## Langfuse layer

Callback wiring lives in `src/agentic_chatbot/observability/callbacks.py`.

When Langfuse keys are set, the live runtime attaches callbacks for:

- chat turns
- upload ingest
- model calls inside general-agent execution
- model calls inside RAG execution
- planner, finalizer, and verifier direct invocations
- worker execution paths launched through the live next runtime

If callback setup fails, the runtime returns an empty callback list and continues.

## Local runtime observability

Runtime artifacts are keyed with `filesystem_key(...)` from
`src/agentic_chatbot_next/runtime/context.py`.

### Session files

```text
data/runtime/sessions/<filesystem_key(session_id)>/
  state.json
  transcript.jsonl
  events.jsonl
  notifications.jsonl
```

### Job files

```text
data/runtime/jobs/<filesystem_key(job_id)>/
  state.json
  transcript.jsonl
  events.jsonl
  mailbox.jsonl
  artifacts/
    output.md
    result.json
```

Related local artifacts:

- `data/workspaces/<filesystem_key(session_id)>/`
- `data/memory/tenants/<tenant>/users/<user>/...`
- `new_demo_notebook/.artifacts/server.log`
- `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`

## What gets recorded

### Session transcript

The session transcript records:

- accepted user messages
- assistant outputs
- task notifications

Task notifications may appear in more than one durable location:

- `notifications.jsonl`
- `state.json` under `pending_notifications`
- `transcript.jsonl` rows with `kind="notification"`

Operational nuance: `notifications.jsonl` can be empty after notification drain. For durable
acceptance triage, also inspect `state.json` and `transcript.jsonl`.

### Runtime events

The runtime emits structured `RuntimeEvent` records for:

- router decisions: `router_decision`
- BASIC-turn lifecycle: `basic_turn_started`, `basic_turn_completed`, `basic_turn_failed`
- AGENT-turn lifecycle: `agent_turn_started`, `agent_turn_completed`, `agent_turn_failed`
- coordinator phases:
  `coordinator_planning_started`, `coordinator_planning_completed`,
  `coordinator_batch_started`, `coordinator_finalizer_completed`,
  `coordinator_verifier_completed`
- worker lifecycle: `worker_agent_started`, `worker_agent_completed`, `worker_agent_failed`
- job lifecycle: `job_created`, `job_started`, `job_completed`, `job_failed`, `job_stopped`
- mailbox lifecycle: `job_message_enqueued`
- callback-driven model lifecycle: `model_start`, `model_end`, `model_error`
- callback-driven tool lifecycle: `tool_start`, `tool_end`, `tool_error`

Each event row carries the persisted `RuntimeEvent` envelope:

- `created_at`
- `session_id`
- `job_id`
- `agent_name`
- `tool_name`
- `payload`

When available, the payload also includes runtime-specific fields such as:

- `conversation_id`
- `route`
- `router_method`
- `suggested_agent`
- coordinator worker and verifier metadata

## Acceptance triage locations

When live acceptance fails, inspect artifacts in this order:

1. `new_demo_notebook/.artifacts/server.log`
2. `data/runtime/sessions/<filesystem_key(session_id)>/events.jsonl`
3. `data/runtime/sessions/<filesystem_key(session_id)>/state.json`
4. `data/runtime/sessions/<filesystem_key(session_id)>/transcript.jsonl`
5. `data/runtime/jobs/<filesystem_key(job_id)>/state.json`
6. `data/runtime/jobs/<filesystem_key(job_id)>/events.jsonl`
7. `data/runtime/jobs/<filesystem_key(job_id)>/artifacts/output.md`
8. `data/runtime/jobs/<filesystem_key(job_id)>/artifacts/result.json`
9. `data/workspaces/<filesystem_key(session_id)>/`
10. `data/memory/tenants/<tenant>/users/<user>/...`

These files are the durable acceptance artifacts for server readiness, worker orchestration,
RAG grounding, data-analyst execution, coordinator job flow, and memory/notification
verification.

## Why both layers exist

Langfuse is useful for centralized trace visualization.

Local runtime files are useful for:

- resume/debug behavior
- worker-job inspection
- auditability when external tracing is unavailable
- development setups where Langfuse is not configured

## Relevant settings

- `LANGFUSE_HOST`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_DEBUG`
- `RUNTIME_EVENTS_ENABLED`
- `RUNTIME_DIR`

## Operational takeaway

If Langfuse is enabled, use it for trace exploration.

If you need the durable local source of truth, inspect the artifacts written by
`src/agentic_chatbot_next/runtime/*`, not the legacy `src/agentic_chatbot/runtime/*`
reference package.
