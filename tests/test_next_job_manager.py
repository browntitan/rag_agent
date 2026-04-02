from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.notification_store import NotificationStore
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def test_job_creation_mailbox_drain_and_notification_persistence(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    notifications = NotificationStore(transcript_store)

    job = job_manager.create_job(
        agent_name="utility",
        prompt="compute",
        session_id="tenant:user:conversation",
        description="utility worker",
        metadata={"test": True},
    )
    assert transcript_store.load_job_state(job.job_id) is not None

    mailbox = job_manager.enqueue_message(job.job_id, "follow up", sender="parent")
    assert mailbox is not None
    drained = job_manager.drain_mailbox(job.job_id)
    assert [item.content for item in drained] == ["follow up"]
    assert job_manager.drain_mailbox(job.job_id) == []

    result = job_manager.run_job_inline(job, lambda current_job: f"done:{current_job.prompt}")
    assert result == "done:compute"

    persisted_job = transcript_store.load_job_state(job.job_id)
    assert persisted_job is not None
    assert persisted_job.status == "completed"
    assert Path(persisted_job.output_path).read_text(encoding="utf-8") == "done:compute"
    assert json.loads(Path(persisted_job.result_path).read_text(encoding="utf-8")) == {"result": "done:compute"}

    notification = job_manager.build_notification(persisted_job)
    notifications.append(persisted_job.session_id, notification)
    drained_notifications = notifications.drain(persisted_job.session_id)
    assert [item.job_id for item in drained_notifications] == [persisted_job.job_id]
    session_notifications_path = paths.session_notifications_path(persisted_job.session_id)
    assert session_notifications_path.read_text(encoding="utf-8") == ""
