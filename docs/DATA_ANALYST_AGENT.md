# Data Analyst Agent

The data analyst agent is a runtime-defined specialist for tabular analysis using Python
in a Docker sandbox.

It is no longer best understood as a graph node. In the live runtime it is an
`AgentDefinition(mode="react")` with a dedicated tool surface and a persistent workspace.

## Overview

| Property | Value |
|---|---|
| Runtime agent name | `data_analyst` |
| Prompt file | `data/skills/data_analyst_agent.md` |
| Runtime mode | `react` |
| Tools | 11 |
| Sandbox | Docker container with bind-mounted workspace |
| Primary file source | indexed KB documents plus persistent session workspace |

## Current tool set

The data analyst runtime agent receives:

1. `load_dataset`
2. `inspect_columns`
3. `execute_code`
4. `calculator`
5. `scratchpad_write`
6. `scratchpad_read`
7. `scratchpad_list`
8. `workspace_write`
9. `workspace_read`
10. `workspace_list`
11. `search_skills`

## Invocation paths

The current orchestrator reaches this agent in two ways:

- directly, when the router suggests `data_analyst`
- indirectly, when another agent spawns it as a worker

## Operating workflow

The agent is still expected to follow a plan-first workflow:

1. inspect the data
2. build an approach
3. execute targeted code
4. verify outputs
5. summarize findings

That behavior comes from the prompt and tool surface, not from a custom graph wrapper.

## Workspace model

The normal execution path now assumes a persistent workspace:

- `data/workspaces/<session_id>/`
- bind-mounted into Docker at `/workspace`

That workspace can now be prepared either by the first chat turn or proactively by
`POST /v1/ingest/documents`, which copies uploaded files into the same session-scoped
directory before the analyst runs.

This means files can survive across turns and across repeated `execute_code` calls.

### Typical flow

1. `load_dataset(doc_id)` resolves a source file from the KB
2. the file path is cached in scratchpad
3. the file is copied into the session workspace when possible
4. `execute_code(...)` runs against the bind-mounted `/workspace`
5. outputs written into `/workspace` persist for the next turn

## Docker behavior

### Isolation properties

| Property | Value |
|---|---|
| Image | `SANDBOX_DOCKER_IMAGE` |
| Network | disabled |
| Timeout | `SANDBOX_TIMEOUT_SECONDS` |
| Memory cap | `SANDBOX_MEMORY_LIMIT` |
| Working directory | `/workspace` |

The container is still ephemeral. Persistence comes from the workspace bind mount, not from
keeping the container alive.

## Fallback file copy behavior

If no persistent workspace is available, `execute_code(...)` falls back to the older
copy-into-container path using Docker file injection. That path is now a fallback, not the
primary design.

## Failure behavior

If Docker is unavailable:

- the `data_analyst` runtime definition still exists
- `execute_code(...)` returns an error payload

So this role is not dynamically removed from the live runtime in the same way the old graph
registry used to hide it.

## Why this agent still matters

The live next runtime uses the same kernel for all AGENT turns, but the data analyst agent
keeps tabular work isolated through:

- a narrower tool surface
- a persistent workspace
- explicit code execution boundaries
- optional skill lookup for analysis procedures
