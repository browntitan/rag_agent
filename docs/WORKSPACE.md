# Session Workspace

The live workspace is a session-scoped host directory used by the data-analyst sandbox and
upload flows.

## Path model

```text
data/workspaces/<filesystem_key(session_id)>/
```

This is intentionally different from the old raw `session_id` directory layout.

## Why it exists

Each Docker sandbox execution is ephemeral, but the workspace bind mount persists across
turns. That lets the runtime:

- copy uploads into `/workspace`
- keep generated files across turns
- let later turns inspect prior outputs

## Live creation paths

The workspace is created by the live next runtime in two common paths:

1. `RuntimeService.process_turn(...)` when a session first needs a workspace
2. `POST /v1/ingest/documents` before the first chat turn, so uploaded files are already
   available in the session-scoped workspace

## Runtime bridge

`SessionState.workspace_root` stores the workspace path. `ToolContext` reconstructs the
workspace handle lazily for tool code.

## What belongs here

- uploaded files copied for analyst/sandbox access
- files written by `workspace_write`
- files created by sandbox code

## What does not belong here

- session state
- transcripts
- events
- job artifacts

Those live under `data/runtime/...`.
