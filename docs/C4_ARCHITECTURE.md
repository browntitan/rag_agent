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
    react["general_agent.py"]
    registry["AgentRegistry"]
    jobs["RuntimeJobManager"]
    tools["Tools / Skills / Memory / RAG"]

    service --> router
    service --> kernel
    kernel --> registry
    kernel --> loop
    kernel --> jobs
    loop --> react
    loop --> tools
    react --> tools
```

## Component notes

- `RuntimeService` is the live service boundary
- `RuntimeKernel` is the persisted session kernel that owns session state, jobs, and notifications
- `QueryLoop` dispatches by agent mode; prompt-backed modes get prompt, memory, and skill
  context, while `rag` and `memory_maintainer` use direct execution paths
- `general_agent.py` is the live react executor for tool-using `react` agents
- `AgentRegistry` loads markdown-defined roles from `data/agents/*.md`
- `RuntimeJobManager` owns durable workers and mailboxes
