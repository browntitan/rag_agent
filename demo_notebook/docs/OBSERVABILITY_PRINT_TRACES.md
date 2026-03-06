# Print Trace Observability Guide

The notebook runtime uses print-based observability only. No Langfuse or containers are required.

## Trace prefixes

- `[NOTEBOOK]`: callback-level LLM/tool lifecycle.
- `[ROUTER]`: deterministic route decisions (BASIC vs AGENT).
- `[GRAPH]`: LangGraph node updates and orchestration flow.
- `[GENERAL_AGENT]`: direct GeneralAgent step/tool call summary.

## What good traces look like

1. Tool-calling flow
- `[NOTEBOOK] Tool start: ...`
- `[NOTEBOOK] Tool end: ...`

2. Graph orchestration flow
- `[GRAPH] node=supervisor updates=next_agent=...`
- `[GRAPH] node=parallel_planner updates=rag_tasks=...`
- `[GRAPH] node=rag_worker updates=worker_results=...`
- `[GRAPH] node=rag_synthesizer updates=final_answer=...`

3. Routing clarity
- `[ROUTER] route=BASIC ...` for direct path
- `[ROUTER] route=AGENT ...` for graph path

## Common failure signatures

- "I could not produce an answer"
  - Usually provider error, prompt failure, or missing retrieval evidence.
- Repeated fallback logs from orchestrator
  - Graph path failed; runtime used `run_general_agent_direct` fallback.
- No tool trace lines in AGENT scenario
  - Model may answer directly; force agent route and use retrieval-heavy prompt.

## Debug checklist

1. Confirm provider mode and credentials in `.env`.
2. Confirm KB is indexed (`Indexed docs: N` output in bootstrap cell).
3. Run parallel scenario with `stream_updates=True` to inspect node progression.
4. Use section F skills showcase to see prompt-control effects with explicit active skill file list.
