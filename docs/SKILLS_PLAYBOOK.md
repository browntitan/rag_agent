# Skills Playbook

This playbook explains how skills work in this codebase, how to author new skills safely, and how to validate skill changes before demos or production-like runs.

## What "skills" mean here

In this repository, a "skill" is a markdown prompt file loaded as system instructions for one or more agents.
Skills are not external plugins. They are instruction files used by the orchestration runtime.

Skill files live in `data/skills/` and are loaded through `src/agentic_chatbot/rag/skills.py`.

## Skill Files and Responsibilities

- `data/skills/skills.md`
  - Shared global context injected into all specialist prompts.
- `data/skills/general_agent.md`
  - Legacy fallback GeneralAgent behavior and tool-use guidance.
- `data/skills/rag_agent.md`
  - RAG agent decision policy (resolve, search, extract, compare, scratchpad).
- `data/skills/supervisor_agent.md`
  - Multi-agent routing policy (`rag_agent`, `utility_agent`, `parallel_rag`, `__end__`).
- `data/skills/utility_agent.md`
  - Utility agent behavior for calculator, doc listing, and memory tools.
- `data/skills/basic_chat.md`
  - Basic no-tool chat behavior.

## Load Timing and Hot Reload Behavior

- `general_agent.md`
  - Loaded at app startup in orchestrator init.
  - Requires app restart to guarantee updates are picked up.
- `basic_chat.md`
  - Loaded at app startup in orchestrator init.
  - Requires app restart for deterministic rollout.
- `rag_agent.md`
  - Loaded on each `run_rag_agent()` call.
  - Changes apply on the next RAG turn.
- `supervisor_agent.md`
  - Loaded when graph nodes are built per turn.
  - Changes apply on the next AGENT turn.
- `utility_agent.md`
  - Loaded when utility node is built per turn.
  - Changes apply on the next AGENT turn.
- `skills.md`
  - Shared include; effective reload behavior follows whichever agent loads it.

## Skill Authoring Rules

Use these rules to keep agent behavior stable and debuggable:

- Be explicit about tool order and stop conditions.
- Use deterministic phrasing for required steps.
- Avoid broad instructions like "use any tool as needed" without guardrails.
- Include failure handling behavior (empty search results, ambiguous doc matches).
- Require transparency when evidence is missing.
- Prefer short imperative statements over long narrative prose.

## Recommended Skill Template

```markdown
# <Agent Name> Instructions

## Mission
<1-2 lines on objective>

## Tool Selection Rules
1. <when to use tool A>
2. <when to use tool B>

## Execution Order
1. <first action>
2. <second action>
3. <synthesis/termination>

## Failure Recovery
- If <condition>: <action>
- If <condition>: <action>

## Output Requirements
- <citation style>
- <format expectations>
- <warning behavior>
```

## Anti-Patterns to Avoid

- Mixing supervisor routing policy into utility skill file.
- Requiring tools unavailable to that agent.
- Conflicting priority rules ("always do X" and "never do X").
- Unbounded retry loops in instructions.
- Hidden assumptions about document names or IDs.

## Adding a New Skill File

1. Create markdown file under `data/skills/`.
2. Add a config path in `.env`/`.env.example` if needed.
3. Wire loader function in `src/agentic_chatbot/rag/skills.py`.
4. Inject into the target runtime path (orchestrator, graph node, or agent).
5. Add docs and demo scenario coverage before rollout.

## Demo Validation Checklist for Skill Changes

Before presenting skill updates:

1. Run `python run.py demo --list-scenarios` to confirm scenario inventory is intact.
2. Run at least one scenario per target agent class:
   - utility
   - rag
   - supervisor/parallel
3. Run with verification enabled:
   - `python run.py demo --scenario <id> --verify --force-agent`
4. Confirm no hard-failure phrases in outputs.
5. If Langfuse is enabled, confirm traces show expected routing and tool calls.

## Production Hardening Suggestions

- Treat skill files like code: PR review, changelog, owner, and rollback plan.
- Maintain a skills regression checklist for high-risk prompts.
- Keep explicit version notes in skill headers for auditability.
- Prefer incremental skill edits over large one-shot rewrites.
