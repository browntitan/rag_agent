# RAG Agent Design

The live RAG path is now the next-runtime contract flow in `src/agentic_chatbot_next/rag/`.

## Live entrypoint

The runtime calls `run_rag_contract(...)`, which returns the preserved RAG contract:

- `answer`
- `citations`
- `used_citation_ids`
- `confidence`
- `retrieval_summary`
- `followups`
- `warnings`

## Current invocation paths

The live next runtime uses the RAG contract flow in:

1. `rag_worker`
2. `rag_agent_tool` exposed to `general` and `verifier`
3. upload-summary kickoff from `RuntimeService.ingest_and_summarize_uploads(...)`

## Current internal stages

The next-owned RAG flow is:

1. retrieve hybrid candidates from PostgreSQL
2. grade candidate chunks for relevance
3. build grounded citations
4. synthesize the answer from bounded evidence
5. coerce the result into the stable contract

Implementation notes:

- `run_rag_contract(...)` currently accepts `skill_context`, `task_context`, and
  `max_retries` for boundary compatibility, but the live implementation does not use those
  inputs yet
- the helper tool factories under `src/agentic_chatbot_next/rag/specialist_tools.py` and
  `src/agentic_chatbot_next/rag/extended_tools.py` exist in the repo, but the live contract
  path does not assemble them

## Corpus model

The live corpus remains DB-first:

- `documents`
- `chunks`
- `collection_id` namespacing

## Why the contract still matters

The stable contract is what lets the runtime reuse the same RAG flow across:

- direct upload summaries
- tool-wrapped usage from `general`
- verifier checks
- dedicated `rag_worker` execution

The contract is stable even while the internal implementation continues to move behind the
next-runtime modules.
