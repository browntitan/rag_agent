# Skills Playbook

The runtime uses two complementary skill layers.

## 1. Base role prompts

Base prompts live in `data/skills/*.md`.

Examples:

- `general_agent.md`
- `rag_agent.md`
- `utility_agent.md`
- `data_analyst_agent.md`
- `planner_agent.md`
- `finalizer_agent.md`
- `supervisor_agent.md` for the opt-in `coordinator`

These files define stable role behavior and should stay compact.

## 2. Retrievable skill packs

Retrievable skill packs live in `data/skill_packs/**/*.md`.

They are:

- file-authored
- version-controlled
- synced into PostgreSQL
- retrieved dynamically through `SkillContextResolver`

## Skill-pack metadata

Supported metadata includes:

- `skill_id`
- `agent_scope`
- `tool_tags`
- `task_tags`
- `version`
- `enabled`
- `description`

## Indexing flow

```mermaid
flowchart LR
    files["data/skill_packs/**/*.md"]
    sync["SkillIndexSync"]
    db["skills + skill_chunks"]
    resolver["SkillContextResolver"]
    runtime["HybridRuntimeLoop"]

    files --> sync --> db --> resolver --> runtime
```

## Runtime flow in the current kernel

For a runtime agent that has `skill_agent_scope` set:

1. the loop resolves bounded skill context for the current user text
2. the context is attached to `ToolContext.skill_context`
3. the base prompt is extended with a `## Skill Context` block
4. the agent runs with targeted guidance for that turn

This replaces the older framing where executor-local prompt assembly was the main way to
inject retrieved skills.

## Where skill retrieval is used

Current runtime roles that normally use retrieved skill context:

- `general`
- `utility`
- `data_analyst`
- `rag_worker`
- `planner`
- `finalizer`
- `verifier`
- `memory_maintainer`

The exact scope is controlled by `AgentDefinition.skill_agent_scope`.

`coordinator` is the main exception in the live runtime: its built-in definition leaves
`skill_agent_scope=""`, so it does not normally receive automatic retrieved skill context.

## Relationship to `search_skills`

Retrieved skill context and `search_skills` are different mechanisms:

- `SkillContextResolver` is automatic, bounded, and kernel-driven
- `search_skills` is a model-invoked tool for explicit lookup during a turn

Both remain useful.

## Commands

Index skill packs:

```bash
python run.py index-skills
```

Inspect indexed skills:

```bash
python run.py list-skills
python run.py inspect-skill <skill_id>
```
