# Control Flow

This document describes the current live turn flow through `agentic_chatbot_next`.

## End-to-end path

```mermaid
flowchart TD
    input["CLI / API input"]
    service["RuntimeService.process_turn()"]
    ingest["optional upload ingest"]
    router["route_turn()"]
    basic["process_basic_turn()"]
    agent["process_agent_turn()"]
    loop["QueryLoop.run()"]
    jobs["worker jobs / notifications"]

    input --> service
    service --> ingest
    ingest --> router
    router -->|"BASIC"| basic
    router -->|"AGENT"| agent
    agent --> loop
    agent --> jobs
```

## 1. Transport layer

The FastAPI gateway and CLI normalize user input, conversation scope, and uploads, then call
`RuntimeService`.

## 2. Service layer

`RuntimeService.process_turn(...)`:

1. opens the session workspace when needed
2. ensures the KB is indexed for the tenant
3. ingests uploads when present
4. calls the router
5. emits a router-decision event
6. hands off to `RuntimeKernel.process_basic_turn(...)` or
   `RuntimeKernel.process_agent_turn(...)`

## 3. BASIC route

`RuntimeKernel.process_basic_turn(...)`:

1. hydrates or creates `SessionState`
2. drains pending notifications
3. appends the user turn
4. persists state and transcript before model execution
5. emits `basic_turn_started`
6. runs the basic chat executor
7. appends the assistant turn
8. persists state and transcript again
9. emits `basic_turn_completed`

## 4. AGENT route

`RuntimeKernel.process_agent_turn(...)`:

1. hydrates or creates `SessionState`
2. drains pending notifications
3. appends the user turn
4. persists state and transcript before execution
5. resolves the initial agent from `data/agents/*.md`
6. builds callbacks and emits `agent_turn_started`
7. delegates to `run_agent(...)`

## 5. Non-coordinator agent execution

`RuntimeKernel.run_agent(...)`:

1. builds `ToolContext`
2. resolves the allowed tool set for the selected agent
3. calls `QueryLoop.run(...)`
4. writes the returned messages back into session state
5. emits completion/failure events

## 6. Coordinator execution

For `coordinator`, the kernel runs:

1. planner
2. worker batching
3. finalizer
4. verifier
5. optional revision pass if verification requests it

Workers run as durable jobs with mailbox continuation and notification reinjection.

## 7. Query loop execution modes

`QueryLoop.run(...)` dispatches by agent mode:

- `basic`
- `react`
- `rag`
- `planner`
- `finalizer`
- `verifier`
- `memory_maintainer`

The loop also injects:

- skill context
- bounded file-memory context

## 8. Persistence timing

The live runtime persists accepted user turns before execution begins.

That means resume/debugging artifacts survive:

- model failures
- tool failures
- worker failures

## 9. Observability

Local runtime events are the source of truth.

Current event families include:

- router decisions
- basic-turn lifecycle
- agent-turn lifecycle
- model lifecycle
- tool lifecycle
- coordinator planning/batch/finalizer/verifier events
- worker-job and mailbox events
- notification append events
- memory extraction events
