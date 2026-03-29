# RAG Agent Design

`run_rag_agent()` is the core document-intelligence engine used across the app.

It can be invoked in three ways:

1. **Primary runtime path (multi-agent graph):** supervisor hands off to `rag_agent` node → evaluator grades output → supervisor loops or ends.
2. **Legacy fallback path:** `GeneralAgent` calls `rag_agent_tool`.
3. **Upload kickoff path:** orchestrator calls `run_rag_agent()` directly after ingest.

---

## Core goals

- high recall across retrieval modes
- grounded answers with inline citations
- robust behavior under tool-budget or model limitations
- stable output contract for upstream synthesis/orchestration

---

## Runtime flow

`run_rag_agent()` (`src/agentic_chatbot/rag/agent.py`) does:

1. Load RAG system prompt from `data/skills/rag_agent.md` (`load_rag_agent_skills`)
2. Build up to 16 specialist RAG tools (`make_all_rag_tools` + optional `make_extended_rag_tools` + optional graph search tools)
3. Try native tool-calling via `create_react_agent`
4. Run ReAct loop until completion or recursion budget
5. Do a separate synthesis call requesting strict JSON output
6. Build/return final contract dict

If the model does not support tool-calling (`bind_tools` fails), it falls back to:

1. `retrieve_candidates()` (vector + keyword)
2. `grade_chunks()` (LLM judge scoring)
3. `generate_grounded_answer()`
4. same contract assembly

---

## The RAG tools (up to 16 active + 5 extended)

Core tools (always available) are built by `make_all_rag_tools()`. Five extended tools are built by `make_extended_rag_tools()`. Two knowledge graph tools are conditionally added when `GRAPHRAG_ENABLED=true`.

### Document navigation

| Tool | Typical use | Return shape |
|---|---|---|
| `resolve_document(name_or_hint)` | resolve human title hints to doc IDs | `{"candidates": [...]}` |
| `list_document_structure(doc_id)` | inspect clause outline before extraction/comparison | `{"doc_id": "...", "outline": [...]}` |
| `search_by_metadata(source_type, file_type, title_contains)` | discover what's indexed before searching content | `[{"doc_id": ..., "title": ...}]` |

### Search tools

| Tool | Strategy options | Typical use |
|---|---|---|
| `search_document(doc_id, query, strategy)` | `hybrid` / `vector` / `keyword` | document-scoped retrieval; returns 500-char snippets + scores |
| `search_all_documents(query, strategy)` | `hybrid` / `vector` / `keyword` | corpus-wide retrieval |
| `full_text_search_document(doc_id, query, max_results)` | PostgreSQL `tsvector` only | when the **complete** chunk text is needed (verbatim quotes, full clauses, tables) |

**Hybrid search scoring:** When `strategy='hybrid'`, results from vector and keyword lists are merged using **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(60 + rank + 1)`. Chunks appearing in both lists are labelled `method='hybrid_rrf'` and naturally ranked higher. This replaces the previous simple highest-score deduplication.

### Extraction tools

| Tool | Backing query | Typical use |
|---|---|---|
| `extract_clauses(doc_id, clause_numbers)` | `WHERE clause_number = ANY(...)` | exact numbered clauses |
| `extract_requirements(doc_id, requirement_filter)` | `WHERE chunk_type='requirement'` (+ optional semantic ranking) | SHALL/MUST/REQ extraction |

### Comparison tools

| Tool | Behavior |
|---|---|
| `compare_clauses(doc_id_1, doc_id_2, clause_numbers)` | returns both doc clause payloads + shared/missing sets |
| `diff_documents(doc_id_1, doc_id_2)` | returns structural overlap + per-doc outline |

### Scratchpad tools

| Tool | Scope |
|---|---|
| `scratchpad_write(key, value, persist=False)` | within-turn; `persist=True` writes to `workspace/.artifacts/<key>.md` for cross-turn retention |
| `scratchpad_read(key)` | reads from memory, falls back to persisted artifact if absent |
| `scratchpad_list()` | lists both in-memory and persisted artifact keys |

### Extended RAG tools (5, from `tools/rag_tools_extended.py`)

| Tool | Purpose |
|---|---|
| `query_rewriter` | Reformulates low-recall queries using a judge LLM |
| `chunk_expander` | Returns surrounding context for a retrieved chunk |
| `document_summarizer` | Generates a concise document summary |
| `citation_validator` | Verifies that a cited chunk ID actually supports a claim |
| `web_search_fallback` | Searches the web via Tavily when KB is insufficient (requires `WEB_SEARCH_ENABLED=true` and `TAVILY_API_KEY`) |

### Knowledge graph tools (2, conditional on `GRAPHRAG_ENABLED=true`)

| Tool | Purpose |
|---|---|
| `graph_search_local` | Entity-level search using Microsoft GraphRAG local search (precise, entity-focused) |
| `graph_search_global` | Community-level search using Microsoft GraphRAG global search (thematic, cross-document) |

---

## Final synthesis contract

After tool execution, synthesis returns/parses this schema:

```json
{
  "answer": "...",
  "used_citation_ids": ["..."],
  "followups": ["..."],
  "warnings": ["..."],
  "confidence_hint": 0.0
}
```

Then `_build_contract(...)` adds:

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
  "confidence": 0.0,
  "retrieval_summary": {
    "query_used": "...",
    "steps": 0,
    "tool_calls_used": 0,
    "tool_call_log": [],
    "citations_found": 0
  },
  "followups": [],
  "warnings": []
}
```

Parse fallback behavior:

- if synthesis JSON parse fails, raw synthesis text is used as `answer`
- warning includes `SYNTHESIS_JSON_PARSE_FAILED`
- fallback confidence defaults low (`0.2` baseline via `confidence_hint`)

---

## Retrieval backend

All retrieval is PostgreSQL-backed through `KnowledgeStores`:

- vector ANN: `chunk_store.vector_search(...)`
- keyword FTS: `chunk_store.keyword_search(...)`
- clause extraction: `chunk_store.get_chunks_by_clause(...)`
- structure outline: `chunk_store.get_structure_outline(...)`
- requirement chunks: `chunk_store.get_requirement_chunks(...)`

---

## Important behavior notes

- `max_retries` is accepted by `run_rag_agent` / `rag_agent_tool` but query rewriting is not currently wired into the active runtime path.
- `rag/rewrite.py` exists as a helper module for future/alternate fallback logic.
- `rag_min_evidence_chunks` is currently configuration-only (not consumed by the active loop).

---

## Ingestion pipeline

Documents pass through this pipeline in order:

1. Load (PyPDF/Docling/OCR) → extract full text
2. Structure detection → classify as `general`, `contract`, `policy_doc`, etc.
3. Clause splitting or fixed-size chunking (based on structure)
4. **Contextual Retrieval** (if `CONTEXTUAL_RETRIEVAL_ENABLED=true`): judge LLM generates a 50-100 token context prefix per chunk, prepended before embedding
5. Embed + store in PostgreSQL (`chunks` table with pgvector HNSW index)
6. **GraphRAG indexing** (if `GRAPHRAG_ENABLED=true`): `graphrag index` runs in a background thread, building a knowledge graph under `GRAPHRAG_DATA_DIR/<doc_id>/`

## Where it lives in code

| Responsibility | File |
|---|---|
| RAG orchestration | `src/agentic_chatbot/rag/agent.py` |
| Core RAG tools (11 base) | `src/agentic_chatbot/tools/rag_tools.py` |
| Extended RAG tools (5) | `src/agentic_chatbot/tools/rag_tools_extended.py` |
| Tool wrapper (`GeneralAgent` fallback path) | `src/agentic_chatbot/tools/rag_agent_tool.py` |
| Retrieval wrappers | `src/agentic_chatbot/rag/retrieval.py` |
| Grading (fallback path) | `src/agentic_chatbot/rag/grading.py` |
| Query rewrite helper (currently not wired) | `src/agentic_chatbot/rag/rewrite.py` |
| Synthesis + citation objects | `src/agentic_chatbot/rag/answer.py` |
| Ingestion + contextual retrieval | `src/agentic_chatbot/rag/ingest.py` |
| Structure detection | `src/agentic_chatbot/rag/structure_detector.py` |
| Clause splitting | `src/agentic_chatbot/rag/clause_splitter.py` |
| OCR | `src/agentic_chatbot/rag/ocr.py` |
| GraphRAG config | `src/agentic_chatbot/graphrag/config.py` |
| GraphRAG indexer | `src/agentic_chatbot/graphrag/indexer.py` |
| GraphRAG searcher | `src/agentic_chatbot/graphrag/searcher.py` |
