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

**How:** `graph/builder.py` builds 9 nodes:

- `supervisor`
- `rag_agent`
- `utility_agent`
- `data_analyst`
- `parallel_planner`
- `rag_worker`
- `rag_synthesizer`
- `evaluator` — Generator-Evaluator pattern (see pattern #17)
- `clarify` — turn-based clarification (see pattern #18)

**Why:** keep tool surfaces and context focused per specialist.

---

## 3. Parallel RAG via Send API with enriched delegation specs

**What:** fan out multiple scoped RAG tasks in parallel with enriched per-task context.

**How:** supervisor sets `rag_sub_tasks`, planner validates and enriches each task with `objective`, `output_format`, `boundary`, and `search_strategy`, workers receive the enriched spec and run `run_rag_agent()`, reducer merges into synthesizer. Vague queries (all tasks < 3 words) trigger clarification instead of fan-out.

**Why:** reduce wall-clock latency for multi-document comparisons; enriched specs prevent workers from duplicating searches and give clear task boundaries.

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

## 10. Scratchpad memory (within-turn + persistent artifacts)

**What:** key-value working memory for RAG, with optional persistence across turns.

**How:** `session.scratchpad` (in-memory dict) + scratchpad tools in `tools/rag_tools.py`. `scratchpad_write(key, value, persist=True)` writes to `workspace/.artifacts/<key>.md` for cross-turn retention. `scratchpad_read` falls back to persisted files if the key is not in memory.

**Why:** support multi-step extraction/comparison workflows within a turn; persist intermediate findings (partial analysis, interim reports) across conversation turns without requiring the full memory stack.

---

## 11. Cross-turn persistent memory

**What:** session-scoped persistent key-value memory.

**How:** PostgreSQL `memory` table via `db/memory_store.py` and `tools/memory_tools.py`.

**Why:** carry user facts across turns/restarts.

---

## 12. Structure-aware ingestion with Contextual Retrieval

**What:** split docs by detected structure, not only fixed size; optionally prepend LLM-generated context to each chunk before embedding.

**How:** `rag/structure_detector.py` + `rag/clause_splitter.py` + ingest dispatcher. When `CONTEXTUAL_RETRIEVAL_ENABLED=true`, `_contextualize_chunks()` calls the judge LLM to generate a 50-100 token context prefix for each chunk summarising its position in the document; the prefix is prepended before embedding.

**Why:** preserve clause semantics for precise extraction/comparison tools. Contextual prefixes reduce retrieval failure rate significantly (Anthropic: ~67% reduction) by giving each embedding vector enough context to be retrieved accurately even for isolated clauses.

---

## 13. Externalized skills prompts

**What:** prompts live in Markdown files.

**How:** `rag/skills.py` loads `data/skills/*.md` via `SkillsLoader` with mtime-based hot-reload and template variable substitution.

**Why:** behavior tuning without code changes.

---

## 14. Dynamic agent registry

**What:** supervisor's knowledge of available agents is derived at runtime, not hardcoded.

**How:** `agents/agent_registry.py` maintains `AgentSpec` objects. `AgentRegistry.format_for_supervisor_prompt()` renders the agent list into the `{{available_agents}}` template variable in `supervisor_agent.md`. `valid_agent_names()` drives JSON validation of supervisor responses.

**Why:** adding a new agent only requires registering it in the registry + wiring graph edges. The supervisor prompt and valid-response enforcement stay in sync automatically. Agents can be conditionally enabled (e.g., `data_analyst` requires Docker).

---

## 15. Isolated Docker sandbox execution

**What:** user-submitted Python code runs in a fresh, network-disabled, memory-limited Docker container per call.

**How:** `sandbox/docker_executor.py` uses the Docker SDK to create a container, copy files via `put_archive()`, execute code, capture stdout/stderr (truncated at 50 KB), and auto-remove the container. Used by `data_analyst` agent via `execute_code` tool.

**Why:** strong isolation prevents code execution from accessing the host network, filesystem, or exceeding memory limits. Graceful degradation when Docker is unavailable.

---

## 16. Plan-verify-reflect data analysis workflow

**What:** structured multi-step data analysis with mandatory inspection before execution and reflection after.

**How:** `data/skills/data_analyst_agent.md` encodes a 5-step workflow (Load → Inspect → Plan → Execute → Verify → Reflect) enforced by the skill prompt. `make_data_analyst_tools()` provides `load_dataset`, `inspect_columns`, `execute_code`, `calculator`, and scratchpad tools.

**Why:** prevents analysis errors from incorrect assumptions about data shape; reflection step ensures the agent validates output before synthesizing a response.

---

## 17. Generator-Evaluator quality gate

**What:** a lightweight LLM evaluation step grades every RAG response against four criteria before it is returned to the user.

**How:** `graph/nodes/evaluator_node.py` runs after `rag_agent`, `utility_agent`, and `rag_synthesizer`. It grades against: relevance (does the answer address the question?), evidence (are chunk citations present?), completeness (are all parts answered?), accuracy (no hallucinated document names?). On failure, it clears `final_answer` and routes back to the supervisor for one retry (`eval_retry_count` guard prevents infinite loops). Utility/data_analyst outputs are passed through unconditionally.

**Why:** catch low-quality or unsupported answers before the user sees them; inspired by Anthropic's harness design for long-running apps (separate generator and evaluator with explicit criteria and loop guard).

---

## 18. Turn-based clarification

**What:** when a request is too vague to route safely, the system asks for more information instead of guessing.

**How:** supervisor sets `next_agent="clarify"` with a `clarification_question`. The `clarify` node (`graph/nodes/clarification_node.py`) emits the question as an `AIMessage` and routes to END. On the next user turn, the answer is in conversation history and the supervisor routes normally. No checkpointer required.

**Why:** prevents wasted agent calls on ambiguous queries; pairs with the parallel planner's vague-query detection (all sub-tasks < 3 words → clarify instead of fan-out).

---

## 19. Microsoft GraphRAG knowledge graph search (opt-in)

**What:** entity extraction and Leiden community detection builds a knowledge graph over ingested documents; local and global search queries the graph for entity-level and community-level answers.

**How:** at ingest time, `_trigger_graphrag_index()` (`rag/ingest.py`) runs `graphrag index` in a background daemon thread via `graphrag/indexer.py`. At query time, `graph_search_local` and `graph_search_global` tools (`tools/rag_tools.py`) call `graphrag query` via `graphrag/searcher.py`. Per-document project directories live under `GRAPHRAG_DATA_DIR`. Enabled via `GRAPHRAG_ENABLED=true` and requires the `graphrag` CLI on PATH.

**Why:** extract implicit relationships between entities (people, organisations, clauses, concepts) that vector search cannot surface; community-level global search provides cross-document thematic summaries.

---

## Summary table

| # | Pattern | Files | Notes |
|---|---|---|---|
| 1 | Router | `router/router.py`, `router/llm_router.py` | deterministic BASIC/AGENT + LLM escalation |
| 2 | Supervisor graph | `graph/builder.py`, `graph/supervisor.py` | multi-agent routing (9 nodes) |
| 3 | Parallel RAG (enriched delegation) | `graph/nodes/rag_worker_node.py`, `graph/nodes/parallel_planner_node.py` | Send API fan-out with delegation specs |
| 4 | ReAct loops | `agents/general_agent.py`, `rag/agent.py` | tool-calling loops |
| 5 | RAG invocation modes | `graph/nodes/rag_node.py`, `tools/rag_agent_tool.py` | handoff primary, tool fallback |
| 6 | Hybrid retrieval (RRF) | `tools/rag_tools.py`, `rag/retrieval.py` | vector + keyword + Reciprocal Rank Fusion |
| 7 | Relevance grading | `rag/grading.py` | fallback-only |
| 8 | Query rewrite helper | `rag/rewrite.py` | present, currently unused in active runtime |
| 9 | Grounded synthesis | `rag/agent.py`, `rag/answer.py` | citation contract |
| 10 | Scratchpad (within-turn + persistent) | `agents/session.py`, `tools/rag_tools.py` | `persist=True` writes to `.artifacts/` |
| 11 | Persistent memory | `db/memory_store.py`, `tools/memory_tools.py` | cross-turn memory |
| 12 | Structured ingest + Contextual Retrieval | `rag/ingest.py`, `rag/structure_detector.py`, `rag/clause_splitter.py` | clause-aware chunks; opt-in LLM context prefix |
| 13 | Skills system | `rag/skills.py`, `rag/skills_loader.py`, `data/skills/*.md` | prompt externalization + hot-reload |
| 14 | Dynamic agent registry | `agents/agent_registry.py`, `data/skills/supervisor_agent.md` | runtime agent discovery |
| 15 | Docker sandbox | `sandbox/docker_executor.py`, `tools/data_analyst_tools.py` | isolated code execution |
| 16 | Plan-verify-reflect | `data/skills/data_analyst_agent.md`, `graph/nodes/data_analyst_node.py` | structured data analysis |
| 17 | Generator-Evaluator | `graph/nodes/evaluator_node.py`, `graph/builder.py` | quality gate; max 1 retry |
| 18 | Clarification node | `graph/nodes/clarification_node.py` | turn-ending clarification |
| 19 | GraphRAG knowledge graph | `graphrag/`, `tools/rag_tools.py` | entity/community search (opt-in) |
