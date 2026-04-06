# Agent Deep Dive

This document describes the live agent system in `agentic_chatbot_next`.

## Entry point

The live entry point is `RuntimeService.process_turn(...)`.

## Agent definitions

Agents are data, not hard-coded classes.

Each role is defined in `data/agents/*.md` and loaded into `AgentDefinition`.

Important fields:

- `mode`
- `prompt_file`
- `allowed_tools`
- `allowed_worker_agents`
- `memory_scopes`
- `max_steps`
- `max_tool_calls`
- `metadata`

## Live execution modes

The next runtime currently uses these agent modes:

- `basic`
- `react`
- `rag`
- `planner`
- `finalizer`
- `verifier`
- `coordinator`
- `memory_maintainer`

## Execution ownership

The runtime splits execution responsibilities across three layers:

- `RuntimeService` handles eager workspace open, upload ingest/upload-summary kickoff,
  routing, and handoff into the kernel
- `RuntimeKernel` owns persisted session state, jobs, notifications, and worker orchestration
- `QueryLoop` dispatches execution by agent mode and injects prompt/skill/memory context

For `react` agents, `QueryLoop` delegates to `agentic_chatbot_next.general_agent.run_general_agent(...)`.
That helper uses LangGraph ReAct when tool binding is available and a plan-execute fallback
when it is not.

Execution-mode nuance:

- `react`, `planner`, `finalizer`, and `verifier` are prompt-backed model executions
- `rag_worker` is a direct `run_rag_contract(...)` call
- `memory_maintainer` is a direct heuristic extractor and does not currently run an LLM or
  ReAct loop

## Role summary

### `basic`

- no tools
- direct chat execution

### `general`

- default AGENT entry
- utility tools
- memory tools
- RAG gateway
- orchestration tools for delegation

### `coordinator`

- manager role for multi-step tasks
- planner/finalizer/verifier orchestration
- worker batching and notifications

### `utility`

- calculator
- document listing
- file-backed memory

### `data_analyst`

- dataset loading
- column inspection
- Docker sandbox execution
- scratchpad and workspace tools

### `rag_worker`

- specialist grounded retrieval path
- returns the preserved RAG contract

### `planner`

- JSON task-plan generator

### `finalizer`

- synthesis over task artifacts

### `verifier`

- output review / revision feedback

### `memory_maintainer`

- explicit delegated helper for writing extracted memory entries into file-backed memory
- separate from the normal post-turn kernel heuristic memory-maintenance path

## Context control

The next runtime uses several boundaries to keep context under control:

- scoped worker prompts
- bounded skill context
- bounded memory context
- session/job transcript separation
- worker mailboxes instead of raw conversation sharing

## Worker execution

Workers are durable jobs, not ad hoc threads of text.

Each worker has:

- a persisted job record
- a mailbox
- transcript rows
- event rows
- task notification output back into the parent session
