from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.persistence.postgres.skills import SkillChunkMatch, SkillPackRecord
from agentic_chatbot_next.persistence.postgres.skills import SkillStore
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.skills.pack_loader import load_skill_pack_from_file


@dataclass
class SkillContext:
    text: str
    matches: List[SkillChunkMatch] = field(default_factory=list)


class SkillIndexSync:
    def __init__(self, settings: Settings, stores: KnowledgeStores) -> None:
        self.settings = settings
        self.stores = stores

    def _skill_packs_root(self) -> Path:
        value = getattr(self.settings, "skill_packs_dir", None)
        return Path(value) if value is not None else Path("data") / "skill_packs"

    def iter_skill_files(self) -> Iterable[Path]:
        root = self._skill_packs_root()
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*.md") if path.is_file())

    def sync(self, *, tenant_id: str) -> Dict[str, Any]:
        indexed: List[Dict[str, Any]] = []
        root = self._skill_packs_root()
        for path in self.iter_skill_files():
            pack = load_skill_pack_from_file(path, root=root)
            self.stores.skill_store.upsert_skill_pack(
                SkillPackRecord(
                    skill_id=pack.skill_id,
                    name=pack.name,
                    agent_scope=pack.agent_scope,
                    checksum=pack.checksum,
                    tenant_id=tenant_id,
                    tool_tags=pack.tool_tags,
                    task_tags=pack.task_tags,
                    version=pack.version,
                    enabled=pack.enabled,
                    source_path=pack.source_path,
                    description=pack.description,
                ),
                pack.chunks,
            )
            indexed.append(
                {
                    "skill_id": pack.skill_id,
                    "name": pack.name,
                    "agent_scope": pack.agent_scope,
                    "chunks": len(pack.chunks),
                    "source_path": pack.source_path,
                }
            )
        return {"indexed": indexed, "count": len(indexed)}


class SkillContextResolver:
    def __init__(self, settings: Settings, stores: KnowledgeStores) -> None:
        self.settings = settings
        self.stores = stores

    def resolve(
        self,
        *,
        query: str,
        tenant_id: str,
        agent_scope: str,
        tool_tags: Optional[List[str]] = None,
        task_tags: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> SkillContext:
        matches = self.stores.skill_store.vector_search(
            query,
            tenant_id=tenant_id,
            top_k=top_k or int(getattr(self.settings, "skill_search_top_k", 4)),
            agent_scope=agent_scope,
            tool_tags=tool_tags,
            task_tags=task_tags,
            enabled_only=True,
        )
        limit = max_chars or int(getattr(self.settings, "skill_context_max_chars", 3000))
        parts: List[str] = []
        consumed = 0
        for match in matches:
            block = f"[{match.name} | {match.skill_id}]\n{match.content.strip()}"
            if consumed + len(block) > limit and parts:
                break
            parts.append(block)
            consumed += len(block) + 2
        return SkillContext(text="\n\n---\n\n".join(parts).strip(), matches=matches)
