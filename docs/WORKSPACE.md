# Session Workspace

## Overview

The **session workspace** is a persistent, host-side directory that is shared by all agents
within a single chat session. It solves the fundamental limitation of the Docker sandbox:
each `execute_code` call previously created a fresh container where all files written by
code disappeared after execution.

```
Host filesystem:
  data/workspaces/<session_id>/
    sales_data.xlsx          ← copied from user upload on ingest
    analysis_results.csv     ← written by execute_code in turn 2
    notes.md                 ← written by agent via workspace_write tool
    report.txt               ← written by execute_code in turn 5
    .artifacts/
      analysis_plan.md       ← written by scratchpad_write(key, value, persist=True)
      interim_findings.md    ← persisted scratchpad artifacts survive turn boundaries

Container (each execute_code call):
  /workspace/  → bind-mounted read-write from data/workspaces/<session_id>/
```

Each Docker container is still **ephemeral and isolated** (network disabled, memory capped,
auto-removed after execution). The persistence comes from bind-mounting the same host
directory into every container — so files written in turn 1 are visible in turn 2 and beyond.

---

## Lifecycle

The workspace is managed automatically:

1. **Open** — on the first call to `process_turn()`, the orchestrator lazily creates the
   workspace directory (`data/workspaces/<session_id>/`) if it does not already exist.
   `open()` is idempotent: if the directory already exists (e.g. from a previous API request
   with the same `conversation_id`), it does nothing.

2. **Use** — all subsequent turns in the same session share the same directory. Files written
   in any turn (via `workspace_write` or `execute_code`'s `print()` to files) persist for the
   next turn.

3. **Close** — the workspace is not automatically closed. Cleanup relies on:
   - **TTL** (default: 24 hours) — set `WORKSPACE_SESSION_TTL_HOURS=0` to disable
   - **Explicit call** — the CLI or a management endpoint can call `session.workspace.close()`
     which runs `shutil.rmtree()` on the workspace root

---

## Tools Available to Agents

The data analyst agent has three workspace tools plus implicit workspace access from
`execute_code`:

### `workspace_write(filename, content)`
Write a text file to the session workspace. Files here survive across turns and are visible
in the sandbox at `/workspace/<filename>`.

```python
workspace_write("analysis_summary.txt", "Region A had 12% higher revenue...")
# → data/workspaces/<session_id>/analysis_summary.txt created
```

### `workspace_read(filename)`
Read a text file from the session workspace. Use `workspace_list()` to discover available files.

```python
content = workspace_read("analysis_summary.txt")
# → returns file content from previous turn
```

### `workspace_list()`
List all files currently in the workspace.

```python
files = workspace_list()
# → {"files": ["analysis_summary.txt", "sales_data.xlsx"], "count": 2}
```

### `execute_code` (implicit)
Python code running in the sandbox has `/workspace/` as its working directory. Files already
in the workspace are available without any special loading:

```python
# This works without calling load_dataset first if the file is already in the workspace:
import pandas as pd
df = pd.read_excel("/workspace/sales_data.xlsx")
```

Code can also write output files that persist after the container is removed:

```python
df_filtered.to_csv("/workspace/filtered_results.csv", index=False)
# → data/workspaces/<session_id>/filtered_results.csv persists for next turn
```

---

## CLI Usage

The CLI creates a `ChatSession` per command invocation. The workspace is keyed by
`session_id`, which defaults to the `conversation_id` from the request context.

For persistent cross-turn workspaces in the CLI, ensure consecutive `ask` or `chat`
commands use the **same conversation ID**. The workspace directory accumulates files
across calls and is cleaned up by TTL (default 24 hours after last use).

---

## API Usage

The API (`/v1/chat/completions`) is stateless — each request creates a fresh `ChatSession`.
To get a persistent workspace across multiple API calls, send the **same
`X-Conversation-ID` header** on every request. The session ID is derived from this header,
so all requests land in the same workspace directory.

```http
POST /v1/chat/completions
X-Conversation-ID: my-project-session-001
Content-Type: application/json

{"model": "...", "messages": [...]}
```

### Multi-turn flow

```
Request 1  (X-Conversation-ID: sess-001)
  orchestrator opens  data/workspaces/sess-001/
  agent writes        data/workspaces/sess-001/plan.txt

Request 2  (X-Conversation-ID: sess-001)
  orchestrator opens  data/workspaces/sess-001/  (idempotent, already exists)
  agent reads         data/workspaces/sess-001/plan.txt  ✓
  execute_code sees   /workspace/plan.txt inside Docker  ✓
```

### File ingestion

Use `/v1/ingest/documents` with `conversation_id` to make uploaded files immediately
available in the workspace sandbox:

```http
POST /v1/ingest/documents
Content-Type: application/json

{
  "paths": ["/path/to/sales_data.xlsx"],
  "source_type": "upload",
  "conversation_id": "sess-001"
}
```

If the workspace directory for `sess-001` already exists on disk (i.e., at least one
`/v1/chat/completions` turn has been processed), the file is copied there automatically.
The response includes `"workspace_copies": ["sales_data.xlsx"]` to confirm.

Alternatively, the `X-Conversation-ID` header is used as a fallback if `conversation_id`
is not in the request body.

---

## Security

- **Docker containers remain ephemeral** — network disabled, memory capped (default 512 MiB),
  auto-removed after execution. The workspace adds persistence, not reduced isolation.
- **Bind-mount is read-write** (`mode: rw`) — code can write files to the workspace.
  Code cannot reach outside `/workspace/` because the container filesystem is otherwise
  isolated.
- **`_safe_filename` validation** — `workspace_write` and `workspace_read` tool calls
  validate filenames against path traversal (`../`), null bytes, and length limits before
  touching the host filesystem.
- **Root directory mode 0o700** — workspace directories are created owner-only.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `WORKSPACE_DIR` | `data/workspaces` | Parent directory for all session workspaces |
| `WORKSPACE_SESSION_TTL_HOURS` | `24` | Hours before a workspace is eligible for TTL cleanup. Set to `0` to disable auto-cleanup. |

These are also available as fields on the `Settings` dataclass:

```python
settings.workspace_dir                # Path object
settings.workspace_session_ttl_hours  # int
```

---

## Scratchpad Artifact Persistence

The RAG agent's `scratchpad_write` tool supports a `persist=True` flag that writes to the `.artifacts/` subdirectory of the workspace:

```python
scratchpad_write(key="my_analysis", value="...", persist=True)
# → workspace/.artifacts/my_analysis.md
```

On subsequent turns, `scratchpad_read("my_analysis")` checks memory first, then falls back to reading from `.artifacts/my_analysis.md`. `scratchpad_list()` includes both in-memory keys and persisted artifact filenames.

This provides lightweight cross-turn file-based handoffs (Anthropic harness pattern) without requiring the full PostgreSQL memory stack.

---

## Architecture Reference

| Component | File | Role |
|---|---|---|
| `SessionWorkspace` | `sandbox/session_workspace.py` | Core class: open/close/read/write/list |
| `workspace_path` param | `sandbox/docker_executor.py` | Bind-mount the workspace into Docker |
| `workspace` field | `agents/session.py` | Attached to `ChatSession` |
| `workspace` field | `graph/session_proxy.py` | Propagated to graph nodes |
| Lazy open + upload copy | `agents/orchestrator.py` | Opens workspace on first turn; copies uploads |
| `workspace_write/read/list` tools | `tools/data_analyst_tools.py` | Agent-callable tools (data analyst) |
| `scratchpad_write(persist=True)` | `tools/rag_tools.py` | RAG agent cross-turn artifact persistence |
| `conversation_id` copy | `api/main.py` | Ingest endpoint workspace sync |
