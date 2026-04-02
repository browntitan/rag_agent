---
name: verifier
mode: verifier
description: Validation-focused worker for checking outputs and citations.
prompt_file: verifier_agent.md
skill_scope: verifier
allowed_tools: ["rag_agent_tool", "list_indexed_docs", "search_skills"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 6
max_tool_calls: 6
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "coordinator_only", "expected_output": "verification_json"}
---
Verifier role definition for the next runtime.
