# Agent Deep Dive

This document describes the live agent system in `agentic_chatbot_next`.

## Entry point

The live entry point is `RuntimeService.process_turn(...)`, not `ChatbotApp`.

`ChatbotApp` still exists only as a deprecated compatibility shim that delegates to
`RuntimeService`.

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

- dedicated maintenance role for writing extracted memory entries into file-backed memory

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
