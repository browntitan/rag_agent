# Observability with Langfuse

This project supports **Langfuse** (via LangChain callbacks) for runtime observability.

When enabled, traces include:

- turn-level metadata (`trace_name`, `session_id`, router route/confidence/reasons)
- multi-agent graph activity (supervisor, utility agent, RAG nodes/synthesizer, evaluator node)
- evaluator LLM calls (grading each RAG response; visible as a separate span)
- fallback `GeneralAgent` tool calls and outputs (when fallback path is used)
- upload ingestion traces (`upload_ingest`)

Note: GraphRAG indexing runs in a background daemon thread and is not instrumented by Langfuse callbacks. GraphRAG query calls (`graph_search_local`, `graph_search_global`) appear in traces as subprocess tool calls within the RAG agent span.

If Langfuse keys are missing or handler setup fails, callbacks are disabled safely.

## Quickstart (local Langfuse)

1) Start Langfuse
2) Create a project
3) Copy keys into `.env`

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_DEBUG=false
```

Then run:

```bash
python run.py chat
```

## Implementation notes

Callback wiring lives in `src/agentic_chatbot/observability/callbacks.py`.
