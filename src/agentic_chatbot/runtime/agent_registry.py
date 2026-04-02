from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agentic_chatbot.runtime.context import AgentDefinition

logger = logging.getLogger(__name__)


_BUILTIN_AGENT_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "general": {
        "name": "general",
        "mode": "react",
        "description": "Default session agent with RAG, utility, and orchestration access.",
        "prompt_key": "general_agent",
        "skill_agent_scope": "general",
        "tool_names": [
            "calculator",
            "list_indexed_docs",
            "memory_save",
            "memory_load",
            "memory_list",
            "rag_agent_tool",
            "search_skills",
            "spawn_worker",
            "message_worker",
            "list_jobs",
            "stop_job",
        ],
        "allowed_worker_agents": ["coordinator", "memory_maintainer"],
        "max_steps": 10,
        "max_tool_calls": 12,
        "allow_background_jobs": True,
        "metadata": {
            "role_kind": "top_level",
            "entry_path": "default",
            "expected_output": "user_text",
            "delegates_complex_tasks_to": "coordinator",
        },
    },
    "coordinator": {
        "name": "coordinator",
        "mode": "coordinator",
        "description": "Opt-in manager agent for explicit worker orchestration.",
        "prompt_key": "supervisor_agent",
        "skill_agent_scope": "",
        "tool_names": ["spawn_worker", "message_worker", "list_jobs", "stop_job"],
        "allowed_worker_agents": ["utility", "rag_worker", "data_analyst", "general", "planner", "finalizer", "verifier", "memory_maintainer"],
        "max_steps": 12,
        "max_tool_calls": 14,
        "allow_background_jobs": True,
        "metadata": {
            "role_kind": "manager",
            "entry_path": "top_level_or_delegated",
            "expected_output": "user_text",
            "planner_agent": "planner",
            "finalizer_agent": "finalizer",
            "verifier_agent": "verifier",
            "verify_outputs": True,
        },
    },
    "utility": {
        "name": "utility",
        "mode": "react",
        "description": "Calculation, document listing, and persistent memory specialist.",
        "prompt_key": "utility_agent",
        "skill_agent_scope": "utility",
        "tool_names": ["calculator", "list_indexed_docs", "memory_save", "memory_load", "memory_list", "search_skills"],
        "max_steps": 8,
        "max_tool_calls": 10,
        "metadata": {
            "role_kind": "worker",
            "entry_path": "delegated",
            "expected_output": "user_text",
        },
    },
    "data_analyst": {
        "name": "data_analyst",
        "mode": "react",
        "description": "Tabular data analysis specialist using sandboxed Python tools.",
        "prompt_key": "data_analyst_agent",
        "skill_agent_scope": "data_analyst",
        "tool_names": [
            "load_dataset",
            "inspect_columns",
            "execute_code",
            "calculator",
            "scratchpad_write",
            "scratchpad_read",
            "scratchpad_list",
            "workspace_write",
            "workspace_read",
            "workspace_list",
            "search_skills",
        ],
        "max_steps": 10,
        "max_tool_calls": 12,
    },
    "rag_worker": {
        "name": "rag_worker",
        "mode": "rag",
        "description": "Grounded document worker that returns the stable RAG contract.",
        "prompt_key": "rag_agent",
        "skill_agent_scope": "rag",
        "tool_names": [],
        "max_steps": 8,
        "max_tool_calls": 12,
        "metadata": {
            "role_kind": "worker",
            "entry_path": "delegated",
            "expected_output": "rag_contract",
        },
    },
    "planner": {
        "name": "planner",
        "mode": "planner",
        "description": "JSON planner for complex multi-step requests.",
        "prompt_key": "planner_agent",
        "skill_agent_scope": "planner",
        "tool_names": [],
        "metadata": {
            "role_kind": "worker",
            "entry_path": "coordinator_only",
            "expected_output": "task_plan_json",
        },
    },
    "finalizer": {
        "name": "finalizer",
        "mode": "finalizer",
        "description": "Final synthesis agent over task artifacts.",
        "prompt_key": "finalizer_agent",
        "skill_agent_scope": "finalizer",
        "tool_names": [],
        "metadata": {
            "role_kind": "worker",
            "entry_path": "coordinator_only",
            "expected_output": "user_text",
        },
    },
    "memory_maintainer": {
        "name": "memory_maintainer",
        "mode": "react",
        "description": "Persistent-memory maintenance helper.",
        "prompt_key": "utility_agent",
        "skill_agent_scope": "utility",
        "tool_names": ["memory_save", "memory_load", "memory_list"],
        "max_steps": 4,
        "max_tool_calls": 4,
        "metadata": {
            "role_kind": "maintenance",
            "entry_path": "delegated",
            "expected_output": "user_text",
        },
    },
    "verifier": {
        "name": "verifier",
        "mode": "verifier",
        "description": "Validation-focused worker for checking outputs and citations.",
        "prompt_key": "verifier_agent",
        "skill_agent_scope": "verifier",
        "tool_names": ["rag_agent_tool", "search_skills", "list_indexed_docs"],
        "max_steps": 6,
        "max_tool_calls": 6,
        "metadata": {
            "role_kind": "worker",
            "entry_path": "coordinator_only",
            "expected_output": "verification_json",
        },
    },
}


class RuntimeAgentRegistry:
    def __init__(self, definitions_dir: Path, *, env_overrides_json: str = "") -> None:
        self.definitions_dir = definitions_dir
        self.env_overrides_json = env_overrides_json
        self._definitions: Dict[str, AgentDefinition] = {}
        self.reload()

    def reload(self) -> None:
        definitions = {
            name: AgentDefinition.from_dict(raw)
            for name, raw in _BUILTIN_AGENT_DEFINITIONS.items()
        }
        for definition in self._iter_file_definitions():
            definitions[definition.name] = definition
        for definition in self._iter_env_definitions():
            definitions[definition.name] = definition
        self._definitions = definitions

    def get(self, name: str) -> Optional[AgentDefinition]:
        return self._definitions.get(name)

    def list(self) -> List[AgentDefinition]:
        return list(self._definitions.values())

    def _iter_file_definitions(self) -> Iterable[AgentDefinition]:
        if not self.definitions_dir.exists():
            return []
        definitions: List[AgentDefinition] = []
        for path in sorted(self.definitions_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                definition = AgentDefinition.from_dict(raw)
                if definition.name:
                    definitions.append(definition)
            except Exception as exc:
                logger.warning("Could not load agent definition from %s: %s", path, exc)
        return definitions

    def _iter_env_definitions(self) -> Iterable[AgentDefinition]:
        if not self.env_overrides_json:
            return []
        try:
            payload = json.loads(self.env_overrides_json)
        except Exception as exc:
            logger.warning("Could not parse AGENT_DEFINITIONS_JSON: %s", exc)
            return []
        if isinstance(payload, dict):
            payload = [payload]
        definitions: List[AgentDefinition] = []
        for item in payload:
            if isinstance(item, dict):
                definition = AgentDefinition.from_dict(item)
                if definition.name:
                    definitions.append(definition)
        return definitions
