from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    return SkillPackFile(
        skill_id=skill_id,
        name=name,
        agent_scope=metadata.get("agent_scope", "rag"),
        body=body or raw,
        chunks=_chunk_skill_body(body or raw),
        checksum=checksum,
        tool_tags=_split_tags(metadata.get("tool_tags", "")),
        task_tags=_split_tags(metadata.get("task_tags", "")),
        version=metadata.get("version", "1"),
        enabled=metadata.get("enabled", "true").lower() not in {"0", "false", "no"},
        source_path=str(path),
        description=metadata.get("description", ""),
    )
