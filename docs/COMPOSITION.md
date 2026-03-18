# How techniques compose in an agentic chatbot

A practical chatbot should not run every expensive technique on every turn.

This repo uses a **composition-by-routing** approach:

1) Router decides `BASIC` vs `AGENT`
2) `AGENT` runs a multi-agent supervisor graph (handoffs to specialist agents)
3) If the graph cannot be built, fallback to legacy `GeneralAgent` tools (including `rag_agent_tool`)
4) Uploads trigger ingestion + immediate RAG kickoff summary

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

- **Primary path:** `rag_agent` is a specialist node in the supervisor graph (agent handoff).
- **Fallback path:** `rag_agent_tool` is called by the legacy `GeneralAgent` as a tool.
- **Upload kickoff:** orchestrator calls `run_rag_agent()` directly after ingest.

## How data analysis is invoked

- **Supervisor handoff:** when the supervisor detects spreadsheet/data analysis intent it routes to `data_analyst`.
- **Tool loop:** `data_analyst` uses `load_dataset` → `inspect_columns` → `execute_code` (Docker sandbox) → reflection.
- **Graceful degradation:** if Docker is not running, `AgentRegistry` disables `data_analyst` at startup and the supervisor never routes to it.

## Upload kickoff

Uploads are a special case:

- ingest immediately
- run a grounded summary for the newly ingested docs
- let the user ask follow-ups

This reduces friction and makes it obvious that the system is using the uploaded docs.
