# How techniques compose in an agentic chatbot

A practical chatbot should not run every expensive technique on every turn.

This repo uses a **composition-by-routing** approach:

1) Router decides `BASIC` vs `AGENT`
2) `AGENT` runs a multi-agent supervisor graph (handoffs to specialist agents)
3) RAG/parallel outputs pass through the **Evaluator Node** (Generator-Evaluator pattern) before returning to supervisor
4) If the graph cannot be built, fallback to legacy `GeneralAgent` tools (including `rag_agent_tool`)
5) Uploads trigger ingestion + immediate RAG kickoff summary

## Recommended defaults

- `BASIC` for:
  - general knowledge
  - small talk

- `AGENT` for:
  - tasks requiring tools (math, memory, doc listing)
  - anything needing grounded document evidence/citations
  - multi-step requests
  - spreadsheet/CSV analysis (routes to `data_analyst`)

## How RAG is invoked

- **Primary path:** `rag_agent` is a specialist node in the supervisor graph (agent handoff) → routes through `evaluator`.
- **Parallel path:** `parallel_planner` fans out to `rag_worker` nodes (enriched delegation specs per worker) → `rag_synthesizer` → `evaluator`.
- **Fallback path:** `rag_agent_tool` is called by the legacy `GeneralAgent` as a tool (bypasses evaluator).
- **Upload kickoff:** orchestrator calls `run_rag_agent()` directly after ingest (bypasses evaluator).

## How data analysis is invoked

- **Supervisor handoff:** when the supervisor detects spreadsheet/data analysis intent it routes to `data_analyst`.
- **Tool loop:** `data_analyst` uses `load_dataset` → `inspect_columns` → `execute_code` (Docker sandbox) → reflection.
- **Graceful degradation:** if Docker is not running, `AgentRegistry` disables `data_analyst` at startup and the supervisor never routes to it.

## How knowledge graph search is invoked (opt-in)

- **Ingest-time:** when `GRAPHRAG_ENABLED=true`, `graphrag index` runs in a background thread after each document is ingested, building entity and community graphs under `GRAPHRAG_DATA_DIR/<doc_id>/`.
- **Query-time:** `rag_agent` gains two additional tools (`graph_search_local`, `graph_search_global`) and can call them when entity or thematic questions require cross-document relationship reasoning.

## How clarification is invoked

- **Supervisor detects ambiguity:** supervisor sets `next_agent="clarify"` with a `clarification_question`. Clarify node emits the question as a response and ends the turn.
- **Parallel planner detects vague queries:** if all sub-tasks in a parallel RAG plan are fewer than 3 words, the planner triggers clarification instead of fan-out.

## Upload kickoff

Uploads are a special case:

- ingest immediately (including contextual retrieval and GraphRAG indexing if enabled)
- run a grounded summary for the newly ingested docs
- let the user ask follow-ups

This reduces friction and makes it obvious that the system is using the uploaded docs.
