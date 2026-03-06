# Skills Guide (Standalone demo_notebook)

This guide documents the notebook-local skills system under `demo_notebook/skills`.

## Purpose

Skills are markdown overlays that modify system prompts for the standalone notebook runtime.
They are used only for demonstration and are intentionally lightweight.

## Skill files

Location: `/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/skills`

- `shared.md`: global prompt constraints used across roles.
- `supervisor.md`: routing priorities and supervisor behavior.
- `rag_agent.md`: evidence and citation behavior for RAG specialist.
- `general_agent.md`: behavior for general ReAct agent.
- `utility_agent.md`: deterministic output style for utility work.
- `skills_showcase_override.md`: high-visibility override used in showcase mode.

## Configuration

In `.env`:

- `NOTEBOOK_SKILLS_ENABLED=true|false`
- `NOTEBOOK_SKILLS_DIR=./skills`
- `NOTEBOOK_SKILLS_SHOWCASE_MODE=true|false`

## Load and composition order

Prompt composition in `runtime/skills.py`:

1. Base prompt (hardcoded runtime default).
2. Shared skill text (`shared.md`).
3. Role skill text (`<role>.md`).
4. Showcase override (`skills_showcase_override.md`) when showcase mode is on.

Final prompt format is additive and does not mutate runtime code paths.

## Scope of impact

When enabled in showcase mode, composed prompts are applied to:

- supervisor node
- RAG agent
- GeneralAgent
- utility specialist prompt
- parallel synthesizer system prompt

When disabled, runtime behavior remains baseline.

## Running the showcase demo

Use notebook section **F) Skills Showcase**.

It runs the same scenario twice:

1. Baseline mode (`skills_showcase_mode=false`)
2. Skills mode (`skills_showcase_mode=true`)

The notebook prints active skill files so prompt inputs are explicit and auditable.

## Authoring tips

- Keep directives short and testable.
- Prefer behavior constraints over broad style instructions.
- Avoid conflicting instructions across shared and role files.
- Use `skills_showcase_override.md` for obvious demo-only behavior deltas.

## Safety note

This skills system is demo-only for this branch deliverable and is isolated from production runtime.
