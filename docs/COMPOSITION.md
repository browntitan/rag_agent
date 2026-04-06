# Composition

The live system is composed around `RuntimeService`.

## Composition order

1. transport layer creates request/session scope
2. `RuntimeService` prepares uploads, workspace, and route metadata
3. `RuntimeKernel` turns the live session into persisted `SessionState`
4. `AgentRegistry` selects the active `AgentDefinition`
5. `QueryLoop` dispatches the selected agent mode and injects shared context
6. tools, worker jobs, notifications, and memory operate inside that runtime context

## Runtime layers

### Service layer

Owns:

- route selection
- eager workspace open plus upload/workspace copy behavior
- upload summary kickoff
- initial agent choice

### Kernel layer

Owns:

- persistence
- events
- jobs
- coordinator orchestration
- notification drain

### Loop layer

Owns:

- prompt construction for prompt-backed modes
- execution dispatch by mode
- handoff to `general_agent.py` for tool-using `react` execution
- direct `run_rag_contract(...)` dispatch for `rag_worker`
- direct heuristic extraction for `memory_maintainer`
- file-memory context injection
- skill-context injection

## Persistence split

- PostgreSQL: documents, chunks, skill embeddings
- `data/runtime`: session/job state, transcripts, events, notifications
- `data/workspaces`: sandbox-visible files
- `data/memory`: file-backed durable memory
