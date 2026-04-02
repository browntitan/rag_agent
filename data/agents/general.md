---
name: general
mode: react
description: Default session agent with RAG, utility, and orchestration access.
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator", "list_indexed_docs", "memory_save", "memory_load", "memory_list", "rag_agent_tool", "search_skills", "spawn_worker", "message_worker", "list_jobs", "stop_job"]
allowed_worker_agents: ["coordinator", "memory_maintainer"]
preload_skill_packs: []
memory_scopes: ["conversation", "user"]
max_steps: 10
max_tool_calls: 12
allow_background_jobs: true
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text", "delegates_complex_tasks_to": "coordinator"}
---
General role definition for the next runtime.
