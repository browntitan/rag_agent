# Architecture

For C4-style context/container/component views, see `docs/C4_ARCHITECTURE.md`.

This system is built around separation of concerns:

1. A deterministic router keeps simple turns cheap (`BASIC`).
2. A multi-agent supervisor graph handles complex turns (`AGENT`).
3. Specialist agents isolate responsibilities (RAG, utility, parallel RAG workers).
4. PostgreSQL (with `pgvector` + `pg_trgm`) is the single backend.
5. A FastAPI OpenAI-compatible gateway (`/v1`) can expose the runtime to external chat UIs.
6. A legacy single-agent fallback is preserved for compatibility.

---

## High-level request flow

```text
User input
  -> Orchestrator (ChatbotApp.process_turn)
  -> Router (BASIC or AGENT)

BASIC route
  -> run_basic_chat (LLM only, no tools)

AGENT route (primary)
  -> build_multi_agent_graph(...)
  -> supervisor
     -> rag_agent
     -> utility_agent
     -> parallel_planner -> rag_worker x N -> rag_synthesizer
     -> __end__

AGENT fallback (capability/config issue only)
  -> GeneralAgent (create_react_agent)
  -> tools: calculator, list_indexed_docs, memory_*, rag_agent_tool

Uploads
  -> ingest_paths(...)
  -> direct run_rag_agent(...) summary kickoff

OpenAI-compatible HTTP gateway (optional)
  -> FastAPI /v1/chat/completions
  -> maps OpenAI messages to ChatSession history + user turn
  -> process_turn(...)

Gateway auth mode:
  -> no in-app auth in simplified mode
  -> secure via network/proxy layer when needed
```

---

## Key design choices

### 1. Deterministic routing before LLM-heavy orchestration

`router/router.py` uses regex heuristics and returns `BASIC`/`AGENT` with confidence + reasons.

Escalation triggers include:

- attachments
- tool/multi-step intent
- citation/grounding language
- high-stakes hints
- long input (`>600` chars)

### 2. Supervisor-driven multi-agent orchestration

The AGENT path uses a LangGraph `StateGraph` (`graph/builder.py`) with six nodes:

- `supervisor`
- `rag_agent`
- `utility_agent`
- `parallel_planner`
- `rag_worker`
- `rag_synthesizer`

The supervisor loops until it chooses `__end__` or loop safety triggers `SUPERVISOR_MAX_LOOPS`.

### 3. RAG invocation modes

RAG runs through the same core function (`run_rag_agent`) in different invocation modes:

- primary: supervisor handoff to `rag_agent` node (`graph/nodes/rag_node.py`)
- parallel: `rag_worker` nodes run independent scoped RAG calls
- upload kickoff: orchestrator direct call after ingestion
- fallback: `rag_agent_tool` called by legacy `GeneralAgent`

### 4. LangGraph ReAct loops (where applicable)

Both `general_agent.py` and `rag/agent.py` use `create_react_agent` when tool-calling is supported.

Benefits:

- graph-managed state
- recursion-budget control
- graceful stop handling
- easier streaming/checkpoint extension path

### 5. PostgreSQL as the single persistence layer

All core data is in PostgreSQL:

- `documents` table for metadata
- `chunks` table for chunk text + embeddings + FTS vector + structure metadata
- `memory` table for session key-value memory

Relevant extensions/indexes:

- `pgvector` (HNSW vector index)
- `pg_trgm` (fuzzy title matching)
- GIN index on generated `tsvector`

### 6. Structure-aware ingestion

`rag/ingest.py` classifies and splits docs using:

- `rag/structure_detector.py`
- `rag/clause_splitter.py`

Document structure types:

- `general`
- `structured_clauses`
- `requirements_doc`
- `policy_doc`
- `contract`

### 7. Skills-driven prompts

Prompt behavior is loaded from `data/skills/*.md` through `rag/skills.py`.

Hot-reload behavior:

- RAG/supervisor/utility prompts are loaded when those nodes are built
- general agent and basic-chat prompts are loaded once in orchestrator init

---

## Memory model

### Scratchpad (within-turn)

- `session.scratchpad` (dict)
- used by RAG scratchpad tools for intermediate findings
- optionally cleared each turn (`CLEAR_SCRATCHPAD_PER_TURN`)

### Persistent memory (cross-turn)

- PostgreSQL `memory` table keyed by `(session_id, key)`
- accessed via `memory_save`, `memory_load`, `memory_list`
- primarily used by `utility_agent` (and fallback `GeneralAgent`)

---

## Fallback behavior

### Multi-agent graph fallback

If graph execution fails due capability/config incompatibility (for example tool-calling support), orchestrator logs warning and runs legacy `GeneralAgent`.
Unexpected graph runtime errors are surfaced explicitly instead of silently masking defects.

### Tool-calling fallback inside agents

If a model cannot `bind_tools`:

- `GeneralAgent`: plan-execute fallback
- `RAGAgent`: retrieval + grading + grounded-answer fallback

Note: `rag/rewrite.py` is present as a helper module but not currently wired into active `run_rag_agent()` flow.

---

## Observability

Langfuse callbacks (when configured) capture:

- turn traces (`chat_turn`, `upload_ingest`)
- router metadata (route/confidence/reasons)
- graph node activity and fallback runs

Callback setup: `src/agentic_chatbot/observability/callbacks.py`

---

## Where patterns appear

| Pattern | Files |
|---|---|
| Router | `router/router.py` |
| Supervisor graph | `graph/builder.py`, `graph/supervisor.py` |
| Parallel RAG | `graph/nodes/parallel_planner_node.py`, `graph/nodes/rag_worker_node.py`, `graph/nodes/rag_synthesizer_node.py` |
| ReAct loops | `agents/general_agent.py`, `rag/agent.py` |
| RAG tool wrapper (fallback path) | `tools/rag_agent_tool.py` |
| Hybrid retrieval | `tools/rag_tools.py`, `rag/retrieval.py` |
| Relevance grading | `rag/grading.py` |
| Grounded synthesis | `rag/answer.py`, `rag/agent.py` |
| Ingestion and structure detection | `rag/ingest.py`, `rag/structure_detector.py`, `rag/clause_splitter.py` |
| OCR ingestion | `rag/ocr.py` |
| Persistent memory | `db/memory_store.py` |
