# Tools and Tool Calling

The live runtime uses tool calling through `src/agentic_chatbot_next`.

## Runtime tool model

The primary runtime abstraction is `ToolDefinition` plus a bound `ToolContext` in
`src/agentic_chatbot_next/tools/`.

Each tool definition carries:

- `name`
- `group`
- `builder`
- `description`
- `args_schema`
- `read_only`
- `destructive`
- `background_safe`
- `concurrency_key`
- `requires_workspace`
- `serializer`
- `metadata`

The registry binds those definitions to the current session/job context and produces
LangChain-compatible tools for agent execution.

Today this surface is intentionally narrower than large coding-agent runtimes:

- tools are Python-defined in-repo
- skills are retrieved prompt context
- agents are markdown-defined roles

The live next runtime does not yet expose a broader plugin or MCP-loaded tool plane.

## Top-level runtime tool groups

### Utility cluster

Exposed through next-runtime tool definitions such as:

- `calculator`
- `list_indexed_docs`
- `memory_save`
- `memory_load`
- `memory_list`
- `search_skills`

In the live runtime these memory tools are file-backed under `data/memory/...`, with
`index.json` as the authoritative store and derived `MEMORY.md` / `topics/*.md` files
for human inspection.

### General cluster

- `rag_agent_tool`

This is how the default `general` agent reaches grounded document reasoning without
receiving the entire internal RAG specialist tool surface directly. The tool fronts the
next-runtime `run_rag_contract()` flow and returns the stable JSON RAG contract.

### Data-analyst cluster

- `load_dataset`
- `inspect_columns`
- `execute_code`
- `scratchpad_write`
- `scratchpad_read`
- `scratchpad_list`
- `workspace_write`
- `workspace_read`
- `workspace_list`

### Orchestration cluster

- `spawn_worker`
- `message_worker`
- `list_jobs`
- `stop_job`

These are only exposed to agents that allow worker orchestration.

## Tool surfaces by runtime agent

### `general`

- utility tools
- `rag_agent_tool`
- orchestration tools

`general` may delegate to `coordinator` or `memory_maintainer` because those worker roles
are explicitly allowed in the live registry.

### `coordinator`

- orchestration tools only

`coordinator` is not a normal ReAct worker. Its runtime mode is `coordinator`, and the
kernel handles planning, task batching, finalization, and optional verification around it.

### `utility`

- calculator
- document listing
- memory tools
- skill search

### `data_analyst`

- dataset inspection
- Docker execution
- scratchpad tools
- workspace tools
- skill search

### `verifier`

- `rag_agent_tool`
- `list_indexed_docs`
- `search_skills`

`verifier` also has its own runtime mode (`verifier`) rather than sharing the generic
`react` path.

### `rag_worker`

No top-level tool exposure. It delegates to the next-runtime RAG contract flow, which uses
a direct Python retrieval/grading/synthesis pipeline.

### `memory_maintainer`

- registry-declared memory tools only

Current implementation note: the dedicated `memory_maintainer` mode bypasses ReAct/tool
calling today and runs direct heuristic extraction in
`QueryLoop._run_memory_maintainer(...)`.

## Additional RAG helper modules

The repo also contains helper tool factories under
`src/agentic_chatbot_next/rag/specialist_tools.py` and
`src/agentic_chatbot_next/rag/extended_tools.py`.

Those modules expose operations such as:

- document resolution
- search across docs or collections
- clause and requirement extraction
- document diff / comparison
- chunk window fetches
- collection listing
- scratchpad helpers
- optional web-search helpers

They are available in the repository, but the live `run_rag_contract()`,
`rag_worker`, and `rag_agent_tool` paths do not currently assemble or invoke them.

## Fallback behavior

If a model wrapper does not support tool calling:

- `run_general_agent()` falls back to a plan-execute loop
- the tool interfaces remain the same from the runtime perspective

This keeps agent behavior functional even when native tool binding is unavailable.

## Safety and metadata

The runtime uses tool metadata primarily for:

- shaping the visible tool surface
- distinguishing read-only vs world-changing operations
- grouping tools by capability
- documenting orchestration permissions through agent config

The current central policy layer is `ToolPolicyService`, which enforces:

- allowed-tool membership per agent
- workspace requirements
- background-job safety
- read-only restrictions for non-effectful modes
- memory-only access for the memory maintainer

For `memory_maintainer`, that policy is mostly defensive today because the live
`memory_maintainer` mode does not execute a ReAct tool loop.

The repo does not yet implement a full human approval layer, which is acceptable for the
current bounded tool surface but should be revisited before adding broader world-changing
capabilities.

## Observability tie-in

Tool execution is now observable through both:

- LangChain callbacks when external tracing is configured
- local `tool_start`, `tool_end`, and `tool_error` events in `data/runtime/*`
