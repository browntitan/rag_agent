from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot.config import Settings
from agentic_chatbot.db.skill_store import SkillChunkMatch, SkillPackRecord
from agentic_chatbot.rag.stores import KnowledgeStores

logger = logging.getLogger(__name__)

_META_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.+?)\s*$")


@dataclass
class SkillPackFile:
    skill_id: str
    name: str
    agent_scope: str
    body: str
    chunks: List[str]
    checksum: str
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    version: str = "1"
    enabled: bool = True
    source_path: str = ""
    description: str = ""


@dataclass
class SkillContext:
    text: str
    matches: List[SkillChunkMatch] = field(default_factory=list)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "skill"


def _split_tags(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _chunk_skill_body(text: str, *, target_chars: int = 900) -> List[str]:
    sections: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
            current_len = len(line)
            continue

        current.append(line)
        current_len += len(line) + 1
        if current_len >= target_chars:
            sections.append("\n".join(current).strip())
            current = []
            current_len = 0

    if current:
        sections.append("\n".join(current).strip())

    return [section for section in sections if section]


def load_skill_pack_from_file(path: Path, *, root: Optional[Path] = None) -> SkillPackFile:
    raw = path.read_text(encoding="utf-8").strip()
    checksum = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
    lines = raw.splitlines()

    name = path.stem.replace("_", " ").replace("-", " ").title()
    metadata: Dict[str, str] = {}
    body_start = 0

    for index, line in enumerate(lines):
        if index == 0 and line.startswith("# "):
            name = line[2:].strip() or name
            continue
        match = _META_PATTERN.match(line)
        if match:
            metadata[match.group(1).strip().lower()] = match.group(2).strip()
            continue
        if not line.strip():
            continue
        body_start = index
        break

    body = "\n".join(lines[body_start:]).strip()
    relative = path.name
    if root is not None:
        try:
            relative = str(path.resolve().relative_to(root.resolve()))
        except Exception:
            relative = path.name
    skill_id = metadata.get("skill_id") or _slugify(relative.replace("/", "-"))
    agent_scope = metadata.get("agent_scope", "rag")
    chunks = _chunk_skill_body(body or raw)

    return SkillPackFile(
        skill_id=skill_id,
        name=name,
        agent_scope=agent_scope,
        body=body or raw,
        chunks=chunks,
        checksum=checksum,
        tool_tags=_split_tags(metadata.get("tool_tags", "")),
        task_tags=_split_tags(metadata.get("task_tags", "")),
        version=metadata.get("version", "1"),
        enabled=metadata.get("enabled", "true").lower() not in {"0", "false", "no"},
        source_path=str(path),
        description=metadata.get("description", ""),
    )


class SkillIndexSync:
    """Synchronize repo-authored skill packs into the DB-backed skill index."""

    def __init__(self, settings: Settings, stores: KnowledgeStores) -> None:
        self.settings = settings
        self.stores = stores

    def iter_skill_files(self) -> Iterable[Path]:
        root = self.settings.skill_packs_dir
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*.md") if path.is_file())

    def sync(self, *, tenant_id: str) -> Dict[str, Any]:
        indexed: List[Dict[str, Any]] = []
        for path in self.iter_skill_files():
            pack = load_skill_pack_from_file(path, root=self.settings.skill_packs_dir)
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
    """Resolve a bounded skill-context block for a task executor."""

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
            top_k=top_k or self.settings.skill_search_top_k,
            agent_scope=agent_scope,
            tool_tags=tool_tags,
            task_tags=task_tags,
            enabled_only=True,
        )

        limit = max_chars or self.settings.skill_context_max_chars
        parts: List[str] = []
        consumed = 0
        for match in matches:
            block = f"[{match.name} | {match.skill_id}]\n{match.content.strip()}"
            if consumed + len(block) > limit and parts:
                break
            parts.append(block)
            consumed += len(block) + 2

        return SkillContext(text="\n\n---\n\n".join(parts).strip(), matches=matches)
