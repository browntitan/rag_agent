# Agent Runtime Guide (`demo_notebook`)

This document explains how notebook agents operate, which model is called at each stage, how demo scenarios execute, and exactly how skills enter LLM context.

## 1) Runtime Overview

Entry point: `DemoOrchestrator` in `demo_notebook/runtime/orchestrator.py`.

Per turn:

1. Router selects `BASIC` or `AGENT`.
2. `BASIC` path sends one chat call with lightweight system prompt.
3. `AGENT` path runs a LangGraph supervisor workflow.
4. If graph execution fails, orchestrator falls back to GeneralAgent direct mode.

## 2) Agent Roles and Responsibilities

### Router (`runtime/router.py`)

- Deterministic classifier (`BASIC` vs `AGENT`), no LLM call.
- `force_agent=True` guarantees `AGENT` route.

### BASIC path (`DemoOrchestrator.run_basic`)

- Single `providers.chat.invoke(...)` call.
- Prompt includes fixed concise system instruction + history + current user turn.

### Supervisor (`runtime/supervisor.py`)

- LLM routing node selecting `rag_agent`, `utility_agent`, `parallel_rag`, or `__end__`.
- Includes JSON parsing fallback and loop guards.

### RAG agent (`runtime/rag_agent.py`)

- ReAct-style tool loop over RAG tools.
- Focuses on retrieval/evidence and synthesis.

### Utility path (`runtime/graph_builder.py` utility node)

- Tool-focused path for calculator/list-doc operations.

### Parallel path (`runtime/graph_builder.py`)

- Planner chooses docs.
- Worker nodes run per-doc analysis.
- Synthesizer merges worker outputs.

### GeneralAgent (`runtime/general_agent.py`)

- ReAct agent used for explicit demo section D and graph fallback.

## 3) Which LLM/Provider Object Is Used Where

Provider bundle fields built in `runtime/providers.py`:

- `providers.chat`
- `providers.judge`
- `providers.embeddings`

Current usage in notebook runtime:

1. KB bootstrap/indexing -> `providers.embeddings`
2. BASIC turn -> `providers.chat`
3. AGENT graph nodes -> `providers.chat`
4. GeneralAgent fallback/direct -> `providers.chat`

Note:

- `providers.judge` is instantiated for parity, but current notebook execution paths primarily use `providers.chat` for generation and orchestration.

## 4) Demo Scenarios and Expected Outputs

Scenario constants live in `runtime/scenarios.py`.

### A) `BASIC_SCENARIO`

- Expect route `BASIC` unless forced.
- No tool-heavy behavior required.

### B) `RAG_CITATION_SCENARIO`

- Expect route `AGENT`.
- Expect RAG tool calls and citation-bearing answer.

### C) `PARALLEL_COMPARE_SCENARIO`

- Expect `parallel_rag` path with planner/worker/synthesizer progression.
- Expect multi-doc synthesis output.

### D) `GENERAL_AGENT_SCENARIO`

- Explicit GeneralAgent run.
- Expect calculator/list-doc behavior and printed step/tool-call stats.

### E) Provider switching notes

- Configuration-only section, not a scenario prompt.

### F) `SKILLS_SHOWCASE_SCENARIO`

- Baseline and skills-enabled runs over same prompt.
- Expect visible style/constraint deltas when skills are enabled.

## 5) Skill Injection Lifecycle

Skills are composed in `runtime/skills.py` and passed by `DemoOrchestrator`.

Activation requirements:

- `NOTEBOOK_SKILLS_ENABLED=true`
- `NOTEBOOK_SKILLS_SHOWCASE_MODE=true`

Composition order:

1. base prompt
2. `skills/shared.md`
3. role skill markdown
4. `skills/skills_showcase_override.md`

### 5.1 Flow table (route -> node -> prompt source -> system message attach point)

| Route | Node/Path | Prompt Source Key | Where `SystemMessage` is attached |
|---|---|---|---|
| BASIC | `run_basic` | fixed basic string (not `SkillProfile`) | `providers.chat.invoke([...])` first message |
| AGENT | Supervisor node | `prompts["supervisor"]` | `make_supervisor_node(..., system_prompt=...)` |
| AGENT | RAG node | `prompts["rag"]` | `run_rag_agent(..., system_prompt=...)` |
| AGENT | Utility node | `prompts["utility"]` | utility path in graph builder |
| AGENT | Parallel synthesizer | `prompts["synthesis"]` | synthesizer invoke `SystemMessage(...)` |
| Fallback / direct | GeneralAgent | `prompts["general"]` | `run_general_agent(..., system_prompt=...)` |

### 5.2 Observable prints during skills mode

When skills mode is active, startup output includes:

- `[NOTEBOOK] skills showcase mode active.`
- `[NOTEBOOK] skill file loaded: <absolute-path>` for each detected file.

These lines confirm actual file selection before scenario execution.

## 6) Observability Signals to Validate Behavior

Use print traces from `runtime/observability.py`:

- `[ROUTER]` for path decisions.
- `[NOTEBOOK] LLM start/end` for model calls.
- `[NOTEBOOK] Tool start/end` for concrete tool execution.
- `[GRAPH]` for node transitions and parallel flow.
- `[GENERAL_AGENT] steps=... tool_calls=...` for direct/fallback runs.

## 7) Why Outputs Can Differ Across Runs

LLM outputs are probabilistic.

Use these stability signals instead of exact phrasing:

1. Route choice (`BASIC` vs `AGENT`).
2. Presence of expected tool calls.
3. Presence of citations for RAG scenario.
4. Absence of loop-limit/failure fallback text.

## 8) Debug Checklist (Skills + Runtime)

1. Confirm provider connectivity and model selection.
2. Confirm KB bootstrap succeeded.
3. Confirm skills gate toggles and loaded file prints.
4. Confirm route aligns with scenario intent (`force_agent` when needed).
5. Confirm expected tools appear in trace output.
