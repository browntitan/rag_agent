# General Delegation Policy
agent_scope: general
tool_tags: rag_agent_tool, spawn_worker, search_skills
task_tags: delegation, orchestration, triage
description: Guidance for when the general agent should solve directly versus delegate.

## Direct execution

- Solve the request directly when the needed work fits inside the current turn and can be completed with your own tools.
- Prefer direct execution for short RAG lookups, simple calculations, document listing, and small memory operations.

## Delegate to the coordinator

- Delegate when the user asks for multi-step research, comparisons across documents, background work, or a task that clearly needs planning and synthesis across scoped workers.
- When delegating, provide a self-contained brief and state any required document scope or artifact references.
