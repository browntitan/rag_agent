# Agent Runtime Guide (demo_notebook)

This document explains how the standalone notebook agents work, which LLM path is used at each step, how demo scenarios execute, and how skills are injected into LLM context.

## 1) Runtime Overview

Entry point: `DemoOrchestrator` in `runtime/orchestrator.py`.

High-level flow per user turn:

1. Router decides `BASIC` vs `AGENT`.
2. `BASIC` route calls direct chat completion.
3. `AGENT` route executes a LangGraph supervisor workflow with specialist agents and tools.
4. If graph execution fails, orchestrator falls back to `GeneralAgent`.

## 2) Agent Roles and Responsibilities

### Router (`runtime/router.py`)

- Type: deterministic rule-based classifier, no LLM.
- Output: `RouteDecision(route, confidence, reasons)`.
- `force_agent=True` always routes to `AGENT`.

### BASIC Assistant Path (`DemoOrchestrator.run_basic`)

- Type: single chat model call.
- Prompt: `"You are a concise assistant."` + conversation history + latest user message.
- Typical use: simple conceptual questions.

### Supervisor (`runtime/supervisor.py`)

- Type: LLM-driven control node.
- Expected output shape: JSON containing next node (`rag_agent`, `utility_agent`, `parallel_rag`, `__end__`).
- Resilience behavior:
  - Invalid JSON falls back to heuristic routing.
  - Deterministic termination guard ends routing once `final_answer` exists.
  - Max loop guard returns `"Stopping after max supervisor loops."` if exceeded.

### RAG Agent (`runtime/rag_agent.py`)

- Type: ReAct agent (`create_react_agent`) with RAG toolset.
- Core behavior: evidence gathering + citation-oriented answer generation.
- Can run globally or document-scoped (`run_rag_agent_for_doc`).

### Utility Agent (`graph_builder.utility_node`)

- Type: specialized `GeneralAgent` run with utility tools only.
- Tool focus: calculator and indexed-doc listing.

### Parallel RAG Path (`runtime/graph_builder.py`)

- `parallel_planner`: chooses up to ~2 docs (or explicit matches in query).
- `rag_worker`: runs document-scoped RAG per selected doc.
- `rag_synthesizer`: merges worker outputs into final consolidated answer.

### GeneralAgent (`runtime/general_agent.py`)

- Type: general ReAct agent.
- Tools: calculator, list docs, and `rag_agent_tool` wrapper.
- Used in two ways:
  - Explicit notebook section D (`run_general_agent_direct`).
  - Orchestrator fallback when graph path raises an exception.

## 3) Which LLM Is Used When

Provider clients are built in `runtime/providers.py` as:
- `providers.chat`
- `providers.judge`
- `providers.embeddings`

Current notebook runtime uses:

1. KB bootstrap/indexing (`bootstrap_kb`) -> `providers.embeddings`
2. BASIC route -> `providers.chat`
3. AGENT route:
- supervisor -> `providers.chat`
- RAG agent -> `providers.chat`
- utility/general agent -> `providers.chat`
- parallel synthesizer -> `providers.chat`
4. Fallback general agent -> `providers.chat`

Important note:
- `providers.judge` is instantiated but not actively used in current demo runtime logic.

## 4) Demo Scenarios and Expected Outputs

Scenario definitions live in `runtime/scenarios.py`.

### A) `BASIC_SCENARIO`

Prompt intent: conceptual explanation (fan-out/fan-in).

Expected behavior:
- Router selects `BASIC` unless forced.
- Output is concise bullet-style explanation.
- No RAG tool requirement.

### B) `RAG_CITATION_SCENARIO`

Prompt intent: policy answer from `09_ai_ops_control_standard.md` with citations.

Expected behavior:
- Usually routed to `AGENT` (or force-agent).
- RAG tools called (e.g., resolve/search/list structure).
- Final answer mentions release-gate and rollback obligations.
- Includes chunk-id style evidence citations.

### C) `PARALLEL_COMPARE_SCENARIO`

Prompt intent: compare v1 vs v2 agreement docs and produce risk board.

Expected behavior:
- AGENT path.
- Supervisor selects `parallel_rag`.
- Planner -> worker fan-out -> synthesizer flow visible in `[GRAPH]` traces.
- Final answer summarizes cross-doc risk deltas with evidence.

### D) `GENERAL_AGENT_SCENARIO`

Prompt intent: list docs by category + perform calculation.

Expected behavior:
- `run_general_agent_direct` path.
- Uses calculator and list-doc capabilities.
- Prints `[GENERAL_AGENT] steps=... tool_calls=...` summary.

### E) Provider Switching Notes

Not a runtime query; this section explains env/provider switching.

### F) `SKILLS_SHOWCASE_SCENARIO`

Prompt intent: executive brief over multiple docs.

Expected behavior:
- Same task run twice:
  - baseline (skills disabled)
  - skills showcase mode enabled
- Skills mode should produce visibly constrained formatting/behavior per override skill file.
- Notebook prints active skill file list.

## 5) Skills: When They Are Used and How They Reach LLM Context

Skills are loaded by `runtime/skills.py` and wired from `DemoOrchestrator`.

Activation conditions:

- `NOTEBOOK_SKILLS_ENABLED=true`
- `NOTEBOOK_SKILLS_SHOWCASE_MODE=true`

If either is false, base prompts are used with no skill overlays.

Prompt composition order:

1. Base prompt (runtime default)
2. `skills/shared.md`
3. Role skill (`skills/supervisor.md`, `skills/rag_agent.md`, etc.)
4. `skills/skills_showcase_override.md` (if enabled)

Injection into LLM context:

- Supervisor: passed as `SystemMessage(content=supervisor_prompt)` in `make_supervisor_node`.
- RAG agent: passed as first `SystemMessage` in `run_rag_agent`.
- GeneralAgent: passed as first `SystemMessage` in `run_general_agent`.
- Utility path: passed as `system_prompt` to `run_general_agent`.
- Parallel synthesis: passed as system prompt in synthesizer invoke.

Traceability:

- When skills mode is active, orchestrator prints each loaded skill path at startup.

## 6) Observability Signals to Verify Correct Behavior

Use print traces from `runtime/observability.py`:

- `[ROUTER]` confirms route and reason.
- `[NOTEBOOK] Tool start/end` confirms actual tool usage.
- `[GRAPH]` confirms supervisor/parallel node transitions.
- `[GENERAL_AGENT]` confirms direct general-agent execution stats.

## 7) Why Outputs May Differ Across Runs

LLM outputs are nondeterministic even with fixed prompts.

Stable indicators to rely on:

1. Route selection (`BASIC` vs `AGENT`).
2. Presence of expected tool calls in traces.
3. Presence of citations for RAG scenarios.
4. Non-empty final answer and no loop-stop error.
