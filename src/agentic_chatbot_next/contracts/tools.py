from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ToolDefinition:
    name: str
    group: str
    builder: Any = ""
    description: str = ""
    args_schema: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = False
    destructive: bool = False
    background_safe: bool = False
    concurrency_key: str = ""
    requires_workspace: bool = False
    serializer: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if callable(self.builder):
            payload["builder"] = getattr(self.builder, "__name__", repr(self.builder))
        return payload

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ToolDefinition":
        return cls(
            name=str(raw.get("name") or ""),
            group=str(raw.get("group") or ""),
            builder=raw.get("builder", ""),
            description=str(raw.get("description") or ""),
            args_schema=dict(raw.get("args_schema") or {}),
            read_only=bool(raw.get("read_only", False)),
            destructive=bool(raw.get("destructive", False)),
            background_safe=bool(raw.get("background_safe", False)),
            concurrency_key=str(raw.get("concurrency_key") or ""),
            requires_workspace=bool(raw.get("requires_workspace", False)),
            serializer=str(raw.get("serializer") or "default"),
            metadata=dict(raw.get("metadata") or {}),
        )
