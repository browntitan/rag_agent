# C4 Architecture

## System context

```mermaid
flowchart TD
    user["User"]
    cli["CLI"]
    api["FastAPI gateway"]
    runtime["agentic_chatbot_next"]
    pg["PostgreSQL + pgvector"]
    disk["data/runtime + data/workspaces + data/memory"]

    user --> cli --> runtime
    user --> api --> runtime
    runtime --> pg
    runtime --> disk
```

## Container view

```mermaid
flowchart TD
    service["RuntimeService"]
    router["Router"]
    kernel["RuntimeKernel"]
    loop["QueryLoop"]
    registry["AgentRegistry"]
    jobs["RuntimeJobManager"]
    tools["Tools / Skills / Memory / RAG"]

    service --> router
    service --> kernel
    kernel --> registry
    kernel --> loop
    kernel --> jobs
    loop --> tools
```

## Component notes

- `RuntimeService` is the live service boundary
- `RuntimeKernel` is the persisted session kernel
- `QueryLoop` is the per-agent execution engine
- `AgentRegistry` loads markdown-defined roles from `data/agents/*.md`
- `RuntimeJobManager` owns durable workers and mailboxes
