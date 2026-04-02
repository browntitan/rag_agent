from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification, WorkerMailboxMessage
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore

logger = logging.getLogger(__name__)

JobRunner = Callable[[JobRecord], str]
TERMINAL_JOB_STATUSES = {"completed", "failed", "stopped"}


class RuntimeJobManager:
    def __init__(
        self,
        transcript_store: RuntimeTranscriptStore,
        *,
        event_sink: Optional[RuntimeEventSink] = None,
        max_worker_concurrency: int = 4,
    ) -> None:
        self.transcript_store = transcript_store
        self.event_sink = event_sink
        self.max_worker_concurrency = max(1, int(max_worker_concurrency))
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(self.max_worker_concurrency)

    def create_job(
        self,
        *,
        agent_name: str,
        prompt: str,
        session_id: str,
        description: str = "",
        session_state: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
        parent_job_id: str = "",
    ) -> JobRecord:
        job = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            agent_name=agent_name,
            status="queued",
            prompt=prompt,
            description=description,
            parent_job_id=parent_job_id,
            session_state=dict(session_state or {}),
            metadata=dict(metadata or {}),
        )
        self.transcript_store.persist_job_state(job)
        self._emit("job_created", job, {"description": description})
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.transcript_store.load_job_state(job_id)

    def list_jobs(self, *, session_id: str = "") -> List[JobRecord]:
        return self.transcript_store.list_job_states(session_id=session_id)

    def start_background_job(self, job: JobRecord, runner: JobRunner) -> JobRecord:
        thread = threading.Thread(target=self._run_job, args=(job.job_id, runner), daemon=True)
        with self._lock:
            self._threads[job.job_id] = thread
        thread.start()
        return job

    def continue_job(self, job_id: str, runner: JobRunner) -> Optional[JobRecord]:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status == "running":
            return job
        return self.start_background_job(job, runner)

    def run_job_inline(self, job: JobRecord, runner: JobRunner) -> str:
        return self._run_job(job.job_id, runner)

    def enqueue_message(self, job_id: str, content: str, *, sender: str = "parent") -> Optional[WorkerMailboxMessage]:
        job = self.get_job(job_id)
        if job is None:
            return None
        message = WorkerMailboxMessage(job_id=job_id, content=content, sender=sender)
        self.transcript_store.append_mailbox_message(message)
        self._emit("mailbox_enqueued", job, {"sender": sender})
        if job.status in {"waiting_message", "failed", "queued"}:
            job.status = "queued"
            job.updated_at = utc_now_iso()
            self.transcript_store.persist_job_state(job)
        return message

    def drain_mailbox(self, job_id: str) -> List[WorkerMailboxMessage]:
        messages = self.transcript_store.load_mailbox_messages(job_id)
        self.transcript_store.overwrite_mailbox(job_id, [])
        return messages

    def stop_job(self, job_id: str) -> Optional[JobRecord]:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.status = "stopped"
        job.updated_at = utc_now_iso()
        self.transcript_store.persist_job_state(job)
        self._emit("job_stopped", job, {})
        return job

    def build_notification(self, job: JobRecord) -> TaskNotification:
        return TaskNotification(
            job_id=job.job_id,
            status=job.status,
            summary=job.result_summary or job.description or f"{job.agent_name} {job.status}",
            output_path=job.output_path,
            result_path=job.result_path,
            result=job.result_summary,
            metadata={"agent_name": job.agent_name},
        )

    def _run_job(self, job_id: str, runner: JobRunner) -> str:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id!r} was not found.")
        if job.status == "stopped":
            return ""
        with self._semaphore:
            job.status = "running"
            job.updated_at = utc_now_iso()
            self.transcript_store.persist_job_state(job)
            self._emit("job_started", job, {})
            try:
                result = runner(job)
                refreshed = self.get_job(job_id) or job
                if refreshed.status == "stopped":
                    return ""
                refreshed.status = "completed"
                refreshed.updated_at = utc_now_iso()
                refreshed.result_summary = result[:2000]
                output_path = self.transcript_store.artifact_path(job_id, "output.md")
                output_path.write_text(result, encoding="utf-8")
                result_path = self.transcript_store.artifact_path(job_id, "result.json")
                result_path.write_text(json.dumps({"result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
                refreshed.output_path = str(output_path)
                refreshed.result_path = str(result_path)
                self.transcript_store.persist_job_state(refreshed)
                self._emit("job_completed", refreshed, {"result_preview": result[:500]})
                return result
            except Exception as exc:
                logger.exception("Background job %s failed", job_id)
                failed = self.get_job(job_id) or job
                failed.status = "failed"
                failed.updated_at = utc_now_iso()
                failed.last_error = str(exc)
                failed.result_summary = str(exc)
                self.transcript_store.persist_job_state(failed)
                self._emit("job_failed", failed, {"error": str(exc)})
                return ""

    def _emit(self, event_type: str, job: JobRecord, payload: Dict[str, object]) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=job.session_id,
                job_id=job.job_id,
                agent_name=job.agent_name,
                payload=dict(payload),
            )
        )
