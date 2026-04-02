# Agent Registry

The live agent registry is `src/agentic_chatbot_next/agents/registry.py`.

## Source of truth

Live agents are loaded from `data/agents/*.md`.

Each file contains markdown frontmatter parsed into `AgentDefinition`:

- `name`
- `mode`
- `description`
- `prompt_file`
- `skill_scope`
- `allowed_tools`
- `allowed_worker_agents`
- `preload_skill_packs`
- `memory_scopes`
- `max_steps`
- `max_tool_calls`
- `allow_background_jobs`
- `metadata`

The old JSON-based runtime agent-loading path is not the live source of truth anymore.

## Load order

`AgentRegistry.reload()`:

1. scans `data/agents/*.md`
2. parses frontmatter through `agentic_chatbot_next.agents.loader`
3. stores definitions by `name`

There is no live built-in-plus-JSON override chain in the next runtime registry.

## Live agent set

Expected live roles:

- `basic`
- `general`
- `coordinator`
- `utility`
- `data_analyst`
- `rag_worker`
- `planner`
- `finalizer`
- `verifier`
- `memory_maintainer`

## How the registry is used

`RuntimeKernel.process_agent_turn(...)`:

1. selects the requested initial agent
2. resolves it through `AgentRegistry`
3. records `active_agent`
4. builds tool policy and execution context from that definition

Coordinator workers are also resolved through the same registry.
