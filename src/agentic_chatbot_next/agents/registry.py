from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agentic_chatbot_next.agents.loader import LoadedAgentFile, load_agent_markdown
from agentic_chatbot_next.contracts.agents import AgentDefinition


class AgentRegistry:
    def __init__(self, agents_dir: Path):
        self.agents_dir = agents_dir
        self._definitions: Dict[str, AgentDefinition] = {}
        self._loaded_files: Dict[str, LoadedAgentFile] = {}
        self.reload()

    def reload(self) -> None:
        definitions: Dict[str, AgentDefinition] = {}
        loaded_files: Dict[str, LoadedAgentFile] = {}
        if self.agents_dir.exists():
            for path in sorted(self.agents_dir.glob("*.md")):
                loaded = load_agent_markdown(path)
                definitions[loaded.definition.name] = loaded.definition
                loaded_files[loaded.definition.name] = loaded
        self._definitions = definitions
        self._loaded_files = loaded_files

    def get(self, name: str) -> Optional[AgentDefinition]:
        return self._definitions.get(name)

    def list(self) -> List[AgentDefinition]:
        return list(self._definitions.values())

    def get_loaded_file(self, name: str) -> Optional[LoadedAgentFile]:
        return self._loaded_files.get(name)
