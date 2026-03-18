# Skills Guide (Standalone `demo_notebook`)

This guide is the canonical technical reference for notebook-local skills in `demo_notebook/skills`.

The notebook skills system is isolated from the production app and only affects the standalone notebook runtime.

## 1) What a Skill Is in `demo_notebook`

A skill is markdown text merged into system prompts. It is not executable code and does not add tools.

- Loader/composer module: `demo_notebook/runtime/skills.py`
- Runtime integration point: `demo_notebook/runtime/orchestrator.py`
- Data container: `SkillProfile(enabled, active_files, prompts)`

## 2) Activation Gate Logic

Skills are only active when both toggles are true:

- `NOTEBOOK_SKILLS_ENABLED=true`
- `NOTEBOOK_SKILLS_SHOWCASE_MODE=true`

The gate check is implemented in `build_skill_profile(...)`:

```python
if not (settings.skills_enabled and settings.skills_showcase_mode):
    return SkillProfile(enabled=False, active_files=[], prompts=dict(base_prompts))
```

If either toggle is false, runtime uses baseline prompts unchanged.

## 3) Skill Files and Scope

Location: `/Users/shivbalodi/Desktop/Rag_Research/langchain_agentic_chatbot_v2/demo_notebook/skills`

| Skill File | Role |
|---|---|
| `shared.md` | Shared constraints across all prompt keys |
| `supervisor.md` | Supervisor routing behavior |
| `rag_agent.md` | RAG agent evidence/citation behavior |
| `general_agent.md` | GeneralAgent behavior |
| `utility_agent.md` | Utility behavior |
| `skills_showcase_override.md` | Extra high-visibility constraints for showcase mode |

## 4) Composition Order and Prompt Model

Prompt composition order is deterministic:

1. Base prompt (runtime default constant)
2. Shared skill text (`shared.md`)
3. Role skill text (`<role>.md`)
4. Showcase override (`skills_showcase_override.md`)

Composed prompt format:

```text
<base prompt>

## Shared Skills
<shared markdown>

## Role Skills
<role markdown>

## Showcase Override
<override markdown>
```

## 5) Prompt Key -> Runtime Node Mapping

`SkillProfile.prompts` contains these keys and targets:

| Prompt Key | Runtime Consumer |
|---|---|
| `supervisor` | Supervisor node (`make_supervisor_node`) |
| `rag` | RAG agent (`run_rag_agent`) |
| `general` | GeneralAgent direct/fallback (`run_general_agent`) |
| `utility` | Utility specialist path in graph builder |
| `synthesis` | Parallel synthesizer prompt in graph builder |

`DemoOrchestrator` passes these prompt values when building graph and direct-agent calls.

## 6) How Section F Proves Skills Behavior

Notebook Section F (`SKILLS_SHOWCASE_SCENARIO`) runs the same task twice:

1. Baseline orchestrator (`skills_enabled=False`, `skills_showcase_mode=False`)
2. Skills orchestrator (`skills_enabled=True`, `skills_showcase_mode=True`)

The notebook prints loaded skill files (`active_skill_files`) when skills mode is active, so prompt inputs are auditable.

## 7) Create a New Notebook Skill (Canonical Workflow)

1. Pick target path first (`supervisor`, `rag`, `general`, `utility`, `synthesis`).
2. Add or edit only the corresponding markdown file.
3. Keep shared policy in `shared.md`; keep role-specific rules in role file.
4. Add explicit and testable rules (tool order, stopping rules, output contract).
5. Re-run Section F and compare baseline vs skills-enabled output.
6. Confirm the changed file appears in printed `active_skill_files`.

Suggested header for each file:

```markdown
# <Skill Name>
Version: 2026-03-06
Owner: <team-or-person>

## Changelog
- 2026-03-06: <what changed and why>
```

## 8) Debugging and Failure Signatures

### Symptom: no behavioral change in skills mode

Checks:

- Confirm both toggles are true.
- Confirm `NOTEBOOK_SKILLS_DIR` points to expected folder.
- Confirm edited file appears in `active_skill_files` printout.

### Symptom: behavior still baseline

Cause:

- Gate disabled (`enabled && showcase_mode` not satisfied).

### Symptom: inconsistent/contradictory outputs

Cause:

- Conflicting directives across `shared.md`, role file, and override.

Fix:

- Remove contradictory "always/never" rules and keep one source of truth per behavior.

### Symptom: wrong tool pattern

Cause:

- Prompt asks for tools that are not in the runtime tool list for that node.

Fix:

- Align skill instructions with actual node tool availability.

## 9) Relationship to Main App Skills

This notebook skills system is branch-local and isolated.

- Notebook skills live in `demo_notebook/skills`.
- Main app skills live in `data/skills` and are loaded by `src/agentic_chatbot/rag/skills.py`.
- Changes in one do not affect the other.
