# Tools and Tool Calling

This codebase uses LangChain tools with LangGraph ReAct agents.

- `utility_agent` (in the multi-agent graph) uses utility tools.
- `run_rag_agent()` uses 11 RAG specialist tools.
- Legacy fallback `GeneralAgent` uses utility tools + `rag_agent_tool`.

---

## Tool design principles

### 1. Narrow tools

A tool should do one thing well.

Good:

- `calculator(expression)` — safe math eval only
- `extract_clauses(doc_id, clause_numbers)` — exact clause retrieval
- `rag_agent_tool(query, ...)` — delegates document intelligence to `run_rag_agent`

Avoid:

- `do_everything(args: dict)`

### 2. Strong descriptions

Tool docstrings are part of the model-facing contract:

- when to use the tool
- argument expectations
- return shape

### 3. Simple schemas

For broad model compatibility (especially local models):

- prefer primitive args (`str`, `int`, `float`, `bool`)
- keep nested JSON payloads simple

### 4. Stable outputs

Tools should return predictable shapes where possible.

---

## ReAct loop pattern

Both `general_agent.py` and `rag/agent.py` use `langgraph.prebuilt.create_react_agent`.

```python
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(llm, tools=tools)
result = graph.invoke(
    {"messages": msgs},
    config={"recursion_limit": budget},
)
```

Budget formulas in code:

```python
# GeneralAgent
recursion_limit = (max(MAX_AGENT_STEPS, MAX_TOOL_CALLS) + 1) * 2 + 1

# RAGAgent
recursion_limit = (MAX_RAG_AGENT_STEPS + MAX_TOOL_CALLS + 1) * 2 + 1
```

---

## Tools in this repo

### Utility tools (calculator + docs + memory)

Used by:

- `utility_agent` in the supervisor graph
- fallback `GeneralAgent`

Tools:

- `calculator` (`tools/calculator.py`)
- `list_indexed_docs` (`tools/list_docs.py`)
- `memory_save`, `memory_load`, `memory_list` (`tools/memory_tools.py`)

### GeneralAgent fallback-only tool

- `rag_agent_tool` (`tools/rag_agent_tool.py`)

This tool wraps `run_rag_agent()` and returns the same contract dict as direct/graph RAG calls.

### RAG specialist tools (11)

Built by `make_all_rag_tools(stores, session)` in `tools/rag_tools.py`:

- `resolve_document`
- `search_document`
- `search_all_documents`
- `extract_clauses`
- `list_document_structure`
- `extract_requirements`
- `compare_clauses`
- `diff_documents`
- `scratchpad_write`
- `scratchpad_read`
- `scratchpad_list`

---

## Key return-shape notes

- `list_document_structure(doc_id)` returns either:
  - `{"doc_id": "...", "outline": [...]}`
  - or a message payload with empty outline when no structure exists.
- `compare_clauses(...)` returns `doc_1_clauses`, `doc_2_clauses`, `missing_in_1`, `missing_in_2`, `shared`.
- `diff_documents(...)` returns `shared`, `only_in_doc_1`, `only_in_doc_2`, plus both outlines.
- `rag_agent_tool(...)` returns a Python dict contract (may be serialized in tool traces).

---

## Tool budgeting and termination

| Parameter | Env Var | Default | Controls |
|---|---|---|---|
| `max_steps` | `MAX_AGENT_STEPS` | 10 | max LLM calls in `GeneralAgent` fallback path |
| `max_tool_calls` | `MAX_TOOL_CALLS` | 12 | max tool invocations in General/RAG loops |
| `max_rag_agent_steps` | `MAX_RAG_AGENT_STEPS` | 8 | max LLM turns in RAG ReAct loop |

When recursion budget is hit, code catches the stop condition and returns a graceful partial response.

---

## Non-tool-calling fallback behavior

If `bind_tools(...)` fails:

- `GeneralAgent`: JSON plan-execute fallback (`_run_plan_execute_fallback`)
- `RAGAgent`: retrieval + grading + grounded answer fallback (no ReAct tool loop)

Files:

- `src/agentic_chatbot/agents/general_agent.py`
- `src/agentic_chatbot/rag/agent.py`
