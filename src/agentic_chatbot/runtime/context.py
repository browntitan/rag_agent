from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeMessage:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    name: str = ""
    tool_call_id: str = ""
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_langchain(self) -> Any:
        kwargs = dict(self.additional_kwargs or {})
        if self.role == "system":
            return SystemMessage(content=self.content, additional_kwargs=kwargs)
        if self.role == "assistant":
            return AIMessage(content=self.content, additional_kwargs=kwargs)
        if self.role == "tool":
            return ToolMessage(
                content=self.content,
                tool_call_id=self.tool_call_id or self.name or "tool_call",
                additional_kwargs=kwargs,
            )
        return HumanMessage(content=self.content, additional_kwargs=kwargs)

    @classmethod
    def from_langchain(cls, message: Any) -> "RuntimeMessage":
        role_map = {
            "human": "user",
            "system": "system",
            "ai": "assistant",
            "tool": "tool",
        }
        role = role_map.get(getattr(message, "type", ""), "assistant")
        tool_call_id = getattr(message, "tool_call_id", "") or ""
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        return cls(
            role=role,
            content=str(content or ""),
            name=str(getattr(message, "name", "") or ""),
            tool_call_id=str(tool_call_id),
            additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskNotification:
    job_id: str
    status: str
    summary: str
    output_file: str = ""
    result: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> RuntimeMessage:
        body = (
            f"<task-notification id=\"{self.job_id}\" status=\"{self.status}\">\n"
            f"{self.summary}\n"
            f"</task-notification>"
        )
        return RuntimeMessage(role="system", content=body, metadata={"notification": asdict(self)})


@dataclass
class WorkerMailboxMessage:
    job_id: str
    content: str
    sender: str = "parent"
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    job_id: str
    agent_name: str
    status: str
    prompt: str
    session_id: str
    description: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    result_summary: str = ""
    output_file: str = ""
    artifact_dir: str = ""
    parent_job_id: str = ""
    last_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDefinition:
    name: str
    mode: str
    description: str = ""
    prompt_key: str = ""
    skill_agent_scope: str = ""
    tool_names: List[str] = field(default_factory=list)
    allowed_worker_agents: List[str] = field(default_factory=list)
    max_steps: int = 10
    max_tool_calls: int = 12
    allow_background_jobs: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AgentDefinition":
        return cls(
            name=str(raw.get("name") or ""),
            mode=str(raw.get("mode") or "react"),
            description=str(raw.get("description") or ""),
            prompt_key=str(raw.get("prompt_key") or ""),
            skill_agent_scope=str(raw.get("skill_agent_scope") or ""),
            tool_names=[str(item) for item in (raw.get("tool_names") or []) if str(item)],
            allowed_worker_agents=[str(item) for item in (raw.get("allowed_worker_agents") or []) if str(item)],
            max_steps=int(raw.get("max_steps") or 10),
            max_tool_calls=int(raw.get("max_tool_calls") or 12),
            allow_background_jobs=bool(raw.get("allow_background_jobs", False)),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ToolBuilder = Callable[["ToolContext"], Iterable[Any]]


@dataclass
class ToolSpec:
    name: str
    builder: ToolBuilder
    description: str = ""
    tags: List[str] = field(default_factory=list)
    read_only: bool = False
    destructive: bool = False
    concurrency_key: str = ""
    availability_rule: str = "always"
    serializer: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    tenant_id: str
    user_id: str
    conversation_id: str
    request_id: str = ""
    session_id: str = ""
    messages: List[RuntimeMessage] = field(default_factory=list)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)
    demo_mode: bool = False
    workspace_root: str = ""
    notifications: List[TaskNotification] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"

    @classmethod
    def from_chat_session(cls, session: Any) -> "SessionState":
        workspace = getattr(session, "workspace", None)
        return cls(
            tenant_id=getattr(session, "tenant_id", "local-dev"),
            user_id=getattr(session, "user_id", "local-cli"),
            conversation_id=getattr(session, "conversation_id", "local-session"),
            request_id=getattr(session, "request_id", ""),
            session_id=getattr(session, "session_id", ""),
            messages=[RuntimeMessage.from_langchain(message) for message in list(getattr(session, "messages", []) or [])],
            uploaded_doc_ids=list(getattr(session, "uploaded_doc_ids", []) or []),
            scratchpad=dict(getattr(session, "scratchpad", {}) or {}),
            demo_mode=bool(getattr(session, "demo_mode", False)),
            workspace_root=str(getattr(workspace, "root", "") or ""),
        )

    def sync_to_chat_session(self, session: Any) -> None:
        session.messages = [message.to_langchain() for message in self.messages]
        session.uploaded_doc_ids = list(self.uploaded_doc_ids)
        session.scratchpad = dict(self.scratchpad)
        session.demo_mode = bool(self.demo_mode)

    def append_message(self, role: str, content: str, **kwargs: Any) -> RuntimeMessage:
        message = RuntimeMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message

    def add_notification(self, notification: TaskNotification) -> None:
        self.notifications.append(notification)
        self.messages.append(notification.to_message())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = [message.to_dict() for message in self.messages]
        payload["notifications"] = [asdict(item) for item in self.notifications]
        return payload

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SessionState":
        state = cls(
            tenant_id=str(raw.get("tenant_id") or "local-dev"),
            user_id=str(raw.get("user_id") or "local-cli"),
            conversation_id=str(raw.get("conversation_id") or "local-session"),
            request_id=str(raw.get("request_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            uploaded_doc_ids=[str(item) for item in (raw.get("uploaded_doc_ids") or []) if str(item)],
            scratchpad={str(k): str(v) for k, v in dict(raw.get("scratchpad") or {}).items()},
            demo_mode=bool(raw.get("demo_mode", False)),
            workspace_root=str(raw.get("workspace_root") or ""),
            metadata=dict(raw.get("metadata") or {}),
        )
        state.messages = [RuntimeMessage(**dict(item)) for item in (raw.get("messages") or []) if isinstance(item, dict)]
        state.notifications = [TaskNotification(**dict(item)) for item in (raw.get("notifications") or []) if isinstance(item, dict)]
        return state


@dataclass
class ToolContext:
    settings: Any
    providers: Any
    stores: Any
    session: SessionState
    callbacks: List[Any] = field(default_factory=list)
    transcript_store: Any = None
    job_manager: Any = None
    event_sink: Any = None
    active_agent: str = ""
    skill_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    _workspace: Any = field(default=None, init=False, repr=False)

    @property
    def workspace_root(self) -> Optional[Path]:
        if not self.session.workspace_root:
            return None
        return Path(self.session.workspace_root)

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def tenant_id(self) -> str:
        return self.session.tenant_id

    @property
    def user_id(self) -> str:
        return self.session.user_id

    @property
    def conversation_id(self) -> str:
        return self.session.conversation_id

    @property
    def request_id(self) -> str:
        return self.session.request_id

    @property
    def uploaded_doc_ids(self) -> List[str]:
        return self.session.uploaded_doc_ids

    @property
    def scratchpad(self) -> Dict[str, str]:
        return self.session.scratchpad

    @property
    def demo_mode(self) -> bool:
        return self.session.demo_mode

    @property
    def workspace(self) -> Any:
        if self._workspace is not None:
            return self._workspace
        root = self.workspace_root
        if root is None:
            return None
        from agentic_chatbot.sandbox.session_workspace import SessionWorkspace  # noqa: PLC0415

        self._workspace = SessionWorkspace.for_session(self.session.session_id, root.parent)
        return self._workspace
