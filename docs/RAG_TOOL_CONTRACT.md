# RAG tool contract

`rag_agent_tool` wraps `run_rag_agent()` and returns the same stable contract schema.

Note: in Python, the tool function returns a dict. In tool-message logs this may appear serialized.

## Input schema

Arguments supported by `rag_agent_tool`:

- `query` (str) — user task/question
- `conversation_context` (str) — optional disambiguation context
- `preferred_doc_ids_csv` (str) — optional comma-separated doc IDs
- `must_include_uploads` (bool)
- `top_k_vector` (int)
- `top_k_keyword` (int)
- `max_retries` (int) — accepted for compatibility; currently not consumed by active runtime logic
- `scratchpad_context_key` (str) — optional key from `session.scratchpad` to prepend context

## Output schema

```json
{
  "answer": "...",
  "citations": [
    {
      "citation_id": "UPLOAD_abcd1234#chunk0003",
      "doc_id": "UPLOAD_abcd1234",
      "title": "my_doc.pdf",
      "source_type": "upload",
      "location": "page 2",
      "snippet": "..."
    }
  ],
  "used_citation_ids": ["UPLOAD_abcd1234#chunk0003"],
  "confidence": 0.84,
  "retrieval_summary": {
    "query_used": "...",
    "steps": 3,
    "tool_calls_used": 5,
    "tool_call_log": ["resolve_document({...})", "search_document({...})"],
    "citations_found": 4
  },
  "followups": ["..."],
  "warnings": []
}
```

## Notes

- `confidence` is adjusted from `confidence_hint` and citation count.
- If synthesis JSON parsing fails, answer falls back to raw text and adds `SYNTHESIS_JSON_PARSE_FAILED` to warnings.
- The same output contract is used when `run_rag_agent()` is called by graph nodes (not only via tool wrapper).
- When called through the multi-agent graph, the output passes through the **Evaluator Node** (`graph/nodes/evaluator_node.py`) before the supervisor decides to end or retry. The evaluator grades: relevance, evidence (citations present), completeness, accuracy. One retry is allowed (`eval_retry_count` guard).

## Scratchpad tool extensions

`scratchpad_write` supports a `persist` flag:

```python
scratchpad_write(key="analysis", value="...", persist=True)
# → writes workspace/.artifacts/analysis.md
# → survives turn and is readable on next turn via scratchpad_read("analysis")
```

`scratchpad_read` falls back to persisted artifact if the key is absent from the in-memory dict.
