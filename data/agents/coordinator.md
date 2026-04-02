---
name: coordinator
mode: coordinator
description: Manager-only role for explicit worker orchestration.
prompt_file: supervisor_agent.md
skill_scope: coordinator
allowed_tools: ["spawn_worker", "message_worker", "list_jobs", "stop_job"]
allowed_worker_agents: ["utility", "rag_worker", "data_analyst", "general", "planner", "finalizer", "verifier", "memory_maintainer"]
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 12
max_tool_calls: 14
allow_background_jobs: true
metadata: {"role_kind": "manager", "entry_path": "delegated_only", "expected_output": "user_text", "planner_agent": "planner", "finalizer_agent": "finalizer", "verifier_agent": "verifier", "verify_outputs": true}
---
Coordinator role definition for the next runtime.
