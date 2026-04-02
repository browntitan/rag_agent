---
name: utility
mode: react
description: Calculation, document listing, and memory specialist.
prompt_file: utility_agent.md
skill_scope: utility
allowed_tools: ["calculator", "list_indexed_docs", "memory_save", "memory_load", "memory_list", "search_skills"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation", "user"]
max_steps: 8
max_tool_calls: 10
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "delegated", "expected_output": "user_text"}
---
Utility role definition for the next runtime.
