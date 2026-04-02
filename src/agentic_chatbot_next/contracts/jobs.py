from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agentic_chatbot_next.contracts.messages import RuntimeMessage, utc_now_iso


@dataclass
class TaskNotification:
    job_id: str
    status: str
    summary: str
    output_path: str = ""
    result_path: str = ""
    result: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_message(self) -> RuntimeMessage:
        body = (
            f"<task-notification id=\"{self.job_id}\" status=\"{self.status}\">\n"
            f"{self.summary}\n"
            f"</task-notification>"
        )
        return RuntimeMessage(
            role="system",
            content=body,
            metadata={"notification": self.to_dict()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskNotification":
        return cls(
            job_id=str(raw.get("job_id") or ""),
            status=str(raw.get("status") or ""),
            summary=str(raw.get("summary") or ""),
            output_path=str(raw.get("output_path") or ""),
            result_path=str(raw.get("result_path") or ""),
            result=str(raw.get("result") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class WorkerMailboxMessage:
    job_id: str
    content: str
    sender: str = "parent"
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerMailboxMessage":
        return cls(
            job_id=str(raw.get("job_id") or ""),
            content=str(raw.get("content") or ""),
            sender=str(raw.get("sender") or "parent"),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    agent_name: str
    status: str
    prompt: str
    description: str = ""
    parent_job_id: str = ""
    artifact_dir: str = ""
    output_path: str = ""
    result_path: str = ""
    result_summary: str = ""
    last_error: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    session_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JobRecord":
        return cls(
            job_id=str(raw.get("job_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            agent_name=str(raw.get("agent_name") or ""),
            status=str(raw.get("status") or "queued"),
            prompt=str(raw.get("prompt") or ""),
            description=str(raw.get("description") or ""),
            parent_job_id=str(raw.get("parent_job_id") or ""),
            artifact_dir=str(raw.get("artifact_dir") or ""),
            output_path=str(raw.get("output_path") or ""),
            result_path=str(raw.get("result_path") or ""),
            result_summary=str(raw.get("result_summary") or ""),
            last_error=str(raw.get("last_error") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            updated_at=str(raw.get("updated_at") or utc_now_iso()),
            session_state=dict(raw.get("session_state") or {}),
            metadata=dict(raw.get("metadata") or {}),
        )
