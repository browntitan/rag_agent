# Empty-Result Recovery And Retry
agent_scope: rag
tool_tags: search_collection, search_document, fetch_chunk_window, search_skills
task_tags: retry, empty results, recovery
version: 1
enabled: true
description: Recover from empty or low-confidence retrieval.

## Recovery Steps

If retrieval returns nothing useful, first change strategy: switch between `hybrid`, `keyword`, and `vector`. Then widen or narrow scope. If you have one promising chunk, use `fetch_chunk_window` to inspect nearby context before concluding the evidence is missing.

## Escalation

After two or three materially different attempts, tell the user what you searched and what was not found. Prefer transparent failure over repeated low-value retries.
