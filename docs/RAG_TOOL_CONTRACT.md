# RAG Tool Contract

`rag_agent_tool` remains backward-compatible.

It wraps `run_rag_agent()` and returns the same stable JSON contract used by the current
live next runtime and its compatibility layers.

## Current positioning

In the live runtime:

- `general` uses `rag_agent_tool` for grounded document work from the top-level ReAct loop
- `verifier` can also use `rag_agent_tool`
- `rag_worker` bypasses the wrapper and calls `run_rag_agent()` directly

So the contract remains central even though not every RAG invocation goes through the tool.

## Input

Supported arguments:

- `query`
- `conversation_context`
- `preferred_doc_ids_csv`
- `must_include_uploads`
- `top_k_vector`
- `top_k_keyword`
- `max_retries`
- `scratchpad_context_key`

## Output

```json
{
  "answer": "...",
  "citations": [
    {
      "citation_id": "...",
      "doc_id": "...",
      "title": "...",
      "source_type": "...",
      "location": "...",
      "snippet": "..."
    }
  ],
  "used_citation_ids": ["..."],
  "confidence": 0.84,
  "retrieval_summary": {
    "query_used": "...",
    "steps": 3,
    "tool_calls_used": 5,
    "tool_call_log": [],
    "citations_found": 4
  },
  "followups": [],
  "warnings": []
}
```

## Stability expectations

The runtime refactor did **not** change:

- key names
- citation object shape
- confidence field
- retrieval summary presence

That stability is what lets the tool remain a safe interface for callers outside the
specialist RAG worker path.

## Related internal RAG tools

The RAG engine still uses a richer internal specialist tool surface for retrieval,
comparison, extraction, and skill lookup. Those tools improve execution quality but do not
change the contract returned by `rag_agent_tool`.
