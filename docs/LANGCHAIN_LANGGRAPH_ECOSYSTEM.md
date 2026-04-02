# LangChain / LangGraph Ecosystem

The live runtime is not “a LangGraph app” at the top level.

## Current split

### LangChain is used for

- chat-model abstractions
- tools
- message types
- callback plumbing

### LangGraph is used for

- the tactical ReAct executor inside `src/agentic_chatbot_next/general_agent.py`

## What is not a LangGraph flow

The following live orchestration is plain Python:

- routing
- session persistence
- coordinator planning/batching/finalization/verification
- worker jobs and mailbox continuation
- notification drain

That logic lives in:

- `src/agentic_chatbot_next/app/service.py`
- `src/agentic_chatbot_next/runtime/kernel.py`
- `src/agentic_chatbot_next/runtime/query_loop.py`

## Direct model-call paths

These modes are currently direct model calls, not LangGraph flows:

- `basic`
- `planner`
- `finalizer`
- `verifier`

The `rag_worker` uses the next-owned RAG retrieval/synthesis path rather than a top-level
LangGraph graph.
