from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot.runtime.context import JobRecord, SessionState, TaskNotification, WorkerMailboxMessage
from agentic_chatbot.runtime.events import RuntimeEvent


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return dict(default or {})
    return json.loads(raw)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


class RuntimeTranscriptStore:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.sessions_root = runtime_root / "sessions"
        self.jobs_root = runtime_root / "jobs"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_root / session_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def persist_session_state(self, session: SessionState) -> None:
        _write_json(self.session_dir(session.session_id) / "state.json", session.to_dict())

    def load_session_state(self, session_id: str) -> Optional[SessionState]:
        path = self.session_dir(session_id) / "state.json"
        if not path.exists():
            return None
        return SessionState.from_dict(_read_json(path))

    def append_session_transcript(self, session_id: str, row: Dict[str, Any]) -> None:
        _append_jsonl(self.session_dir(session_id) / "transcript.jsonl", row)

    def append_session_event(self, event: RuntimeEvent) -> None:
        _append_jsonl(self.session_dir(event.session_id) / "events.jsonl", event.to_dict())

    def session_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "transcript.jsonl")

    def session_events(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "events.jsonl")

    def append_session_notification(self, session_id: str, notification: TaskNotification) -> None:
        _append_jsonl(self.session_dir(session_id) / "notifications.jsonl", notification.__dict__)

    def session_notifications(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "notifications.jsonl")

    def persist_job_state(self, job: JobRecord) -> None:
        job_dir = self.job_dir(job.job_id)
        artifact_dir = job_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if not job.artifact_dir:
            job.artifact_dir = str(artifact_dir)
        _write_json(job_dir / "state.json", job.to_dict())

    def load_job_state(self, job_id: str) -> Optional[JobRecord]:
        path = self.job_dir(job_id) / "state.json"
        if not path.exists():
            return None
        return JobRecord(**_read_json(path))

    def list_job_states(self, *, session_id: str = "") -> List[JobRecord]:
        jobs: List[JobRecord] = []
        for path in sorted(self.jobs_root.glob("*/state.json")):
            raw = _read_json(path)
            if session_id and str(raw.get("session_id")) != session_id:
                continue
            jobs.append(JobRecord(**raw))
        jobs.sort(key=lambda item: item.created_at)
        return jobs

    def append_job_transcript(self, job_id: str, row: Dict[str, Any]) -> None:
        _append_jsonl(self.job_dir(job_id) / "transcript.jsonl", row)

    def append_job_event(self, event: RuntimeEvent) -> None:
        _append_jsonl(self.job_dir(event.job_id) / "events.jsonl", event.to_dict())

    def job_events(self, job_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.job_dir(job_id) / "events.jsonl")

    def append_mailbox_message(self, message: WorkerMailboxMessage) -> None:
        _append_jsonl(self.job_dir(message.job_id) / "mailbox.jsonl", message.to_dict())

    def load_mailbox_messages(self, job_id: str) -> List[WorkerMailboxMessage]:
        return [WorkerMailboxMessage(**row) for row in _read_jsonl(self.job_dir(job_id) / "mailbox.jsonl")]

    def overwrite_mailbox(self, job_id: str, messages: Iterable[WorkerMailboxMessage]) -> None:
        path = self.job_dir(job_id) / "mailbox.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def artifact_path(self, job_id: str, filename: str) -> Path:
        safe_name = filename.replace("/", "_")
        artifacts = self.job_dir(job_id) / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return artifacts / safe_name
