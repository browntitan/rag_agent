# Agentic patterns implemented

This repo implements a production-oriented multi-agent document intelligence architecture.

---

## 1. Routing: cheap vs expensive path

**What:** deterministic route between `BASIC` and `AGENT`.

**How:** regex router in `router/router.py`.

**Why:** avoid paying agent/tool overhead for simple turns.

---

## 2. Supervisor multi-agent graph (LangGraph `StateGraph`)

**What:** LLM supervisor routes to specialist nodes.

**How:** `graph/builder.py` builds 6 nodes:

- `supervisor`
- `rag_agent`
- `utility_agent`
- `parallel_planner`
- `rag_worker`
- `rag_synthesizer`

**Why:** keep tool surfaces and context focused per specialist.

---

## 3. Parallel RAG via Send API

**What:** fan out multiple scoped RAG tasks in parallel.

**How:** supervisor sets `rag_sub_tasks`, planner validates, workers run `run_rag_agent()`, reducer merges into synthesizer.

**Why:** reduce wall-clock latency for multi-document comparisons.

---

## 4. ReAct tool-calling loops (`create_react_agent`)

**What:** iterative LLM→tools→LLM loop.

**How:** used in `agents/general_agent.py` and `rag/agent.py` when tool calling is supported.

**Why:** robust multi-step tool orchestration with built-in recursion budget control.

---

## 5. RAG as specialist node (primary) + tool wrapper (fallback)

**What:** primary AGENT path uses agent handoff to `rag_agent`; fallback path uses `rag_agent_tool`.

**How:**

- primary: `graph/nodes/rag_node.py` invokes `run_rag_agent()`
- fallback: `tools/rag_agent_tool.py` wraps `run_rag_agent()` for legacy `GeneralAgent`

**Why:** modern path gets clean multi-agent orchestration; legacy compatibility remains intact.

---

## 6. Hybrid retrieval fan-out/fan-in

**What:** vector + keyword retrieval, then dedupe.

**How:** active RAG tools in `tools/rag_tools.py` run vector+keyword and dedupe by chunk ID; `rag/retrieval.py` provides the same pattern for non-tool-calling fallback.

**Why:** better recall than vector-only or keyword-only.

---

## 7. LLM-as-a-judge relevance grading (fallback path)

**What:** score chunks 0-3 and keep stronger evidence.

**How:** `rag/grading.py`.

**Why:** improve precision when running non-tool-calling RAG fallback.

---

## 8. Query-rewrite helper module (currently not wired into active runtime)

**What:** rewrite utility for retrieval query reformulation.

**How:** `rag/rewrite.py`.

**Why:** available for future/alternate fallback logic; not currently called by `run_rag_agent()`.

---

## 9. Grounded synthesis with citation contract

**What:** answer must trace to retrieved evidence.

**How:** final synthesis in `rag/agent.py` + citation construction in `rag/answer.py`.

**Why:** auditable document QA and safer enterprise usage.

---

## 10. Within-turn scratchpad memory

**What:** turn-scoped key-value working memory for RAG.

**How:** `session.scratchpad` + scratchpad tools in `tools/rag_tools.py`.

**Why:** support multi-step extraction/comparison workflows.

---

## 11. Cross-turn persistent memory

**What:** session-scoped persistent key-value memory.

**How:** PostgreSQL `memory` table via `db/memory_store.py` and `tools/memory_tools.py`.

**Why:** carry user facts across turns/restarts.

---

## 12. Structure-aware ingestion

**What:** split docs by detected structure, not only fixed size.

**How:** `rag/structure_detector.py` + `rag/clause_splitter.py` + ingest dispatcher.

**Why:** preserve clause semantics for precise extraction/comparison tools.

---

## 13. Externalized skills prompts

**What:** prompts live in Markdown files.

**How:** `rag/skills.py` loads `data/skills/*.md`.

**Why:** behavior tuning without code changes.

---

## Summary table

| # | Pattern | Files | Notes |
|---|---|---|---|
| 1 | Router | `router/router.py` | deterministic BASIC/AGENT |
| 2 | Supervisor graph | `graph/builder.py`, `graph/supervisor.py` | multi-agent routing |
| 3 | Parallel RAG | `graph/nodes/rag_worker_node.py` | Send API fan-out |
| 4 | ReAct loops | `agents/general_agent.py`, `rag/agent.py` | tool-calling loops |
| 5 | RAG invocation modes | `graph/nodes/rag_node.py`, `tools/rag_agent_tool.py` | handoff primary, tool fallback |
| 6 | Hybrid retrieval | `tools/rag_tools.py`, `rag/retrieval.py` | vector + keyword |
| 7 | Relevance grading | `rag/grading.py` | fallback-only |
| 8 | Query rewrite helper | `rag/rewrite.py` | present, currently unused |
| 9 | Grounded synthesis | `rag/agent.py`, `rag/answer.py` | citation contract |
| 10 | Scratchpad | `agents/session.py`, `tools/rag_tools.py` | within-turn memory |
| 11 | Persistent memory | `db/memory_store.py`, `tools/memory_tools.py` | cross-turn memory |
| 12 | Structured ingest | `rag/ingest.py`, `rag/structure_detector.py`, `rag/clause_splitter.py` | clause-aware chunks |
| 13 | Skills system | `rag/skills.py`, `data/skills/*.md` | prompt externalization |
