---
name: data_analyst
mode: react
description: Tabular data analysis specialist using sandboxed Python tools.
prompt_file: data_analyst_agent.md
skill_scope: data_analyst
allowed_tools: ["load_dataset", "inspect_columns", "execute_code", "calculator", "scratchpad_write", "scratchpad_read", "scratchpad_list", "workspace_write", "workspace_read", "workspace_list", "search_skills"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 10
max_tool_calls: 12
allow_background_jobs: false
metadata: {"role_kind": "top_level_or_worker", "entry_path": "router_fast_path_or_delegated", "expected_output": "user_text", "execution_strategy": "plan_execute"}
---
Data analyst role definition for the next runtime.
