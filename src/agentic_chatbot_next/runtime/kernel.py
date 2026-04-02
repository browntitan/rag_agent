from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_chatbot_next.agents.prompt_builder import PromptBuilder
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.observability.callbacks import (
    RuntimeTraceCallbackHandler,
    get_langchain_callbacks,
)
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.event_sink import NullEventSink, RuntimeEventSink
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.notification_store import NotificationStore
from agentic_chatbot_next.runtime.query_loop import QueryLoop, QueryLoopResult
from agentic_chatbot_next.runtime.task_plan import (
    TERMINAL_TASK_STATUSES,
    TaskExecutionState,
    TaskResult,
    VerificationResult,
    WorkerExecutionRequest,
    select_execution_batch,
)
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.skills.runtime import SkillRuntime
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.tools.executor import build_agent_tools
from agentic_chatbot_next.tools.policy import ToolPolicyService
from agentic_chatbot_next.tools.registry import build_tool_definitions
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


class TranscriptEventSink(RuntimeEventSink):
    def __init__(self, transcript_store: RuntimeTranscriptStore) -> None:
        self.transcript_store = transcript_store

    def emit(self, event: RuntimeEvent) -> None:
        self.transcript_store.append_session_event(event)
        if event.job_id:
            self.transcript_store.append_job_event(event)


@dataclass
class AgentRunResult:
    text: str
    messages: List[RuntimeMessage]
    metadata: Dict[str, Any]


class RuntimeKernel:
    def __init__(
        self,
        settings: Any,
        providers: Any | None = None,
        stores: Any | None = None,
        *,
        paths: Optional[RuntimePaths] = None,
        registry: Optional[AgentRegistry] = None,
        query_loop: Optional[QueryLoop] = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self.paths = paths or RuntimePaths.from_settings(settings)
        self.transcript_store = RuntimeTranscriptStore(self.paths)
        if getattr(settings, "runtime_events_enabled", True):
            event_sink: RuntimeEventSink = TranscriptEventSink(self.transcript_store)
        else:
            event_sink = NullEventSink()
        self.event_sink = event_sink
        self.notification_store = NotificationStore(self.transcript_store)
        self.job_manager = RuntimeJobManager(
            self.transcript_store,
            event_sink=self.event_sink,
            max_worker_concurrency=int(getattr(settings, "max_worker_concurrency", 4)),
        )
        agents_dir = Path(getattr(settings, "agents_dir", Path("data") / "agents"))
        self.registry = registry or AgentRegistry(agents_dir)
        self.prompt_builder = PromptBuilder(Path(getattr(settings, "skills_dir", Path("data") / "skills")))
        self.skill_runtime = SkillRuntime(settings, stores, self.prompt_builder) if stores is not None else None
        self.query_loop = query_loop or QueryLoop(
            settings=settings,
            providers=providers,
            stores=stores,
            skill_runtime=self.skill_runtime,
        )
        self.tool_policy = ToolPolicyService()
        self.file_memory_store = FileMemoryStore(self.paths)
        self.memory_extractor = MemoryExtractor(self.file_memory_store)
        self._validate_registry()

    def hydrate_session_state(self, session: Any) -> SessionState:
        state = self._load_or_build_session_state(session)
        self._drain_pending_notifications(state)
        return state

    def emit_router_decision(
        self,
        session: Any,
        *,
        route: str,
        confidence: float,
        reasons: List[str],
        router_method: str,
        suggested_agent: str,
        force_agent: bool,
        has_attachments: bool,
    ) -> None:
        conversation_id = str(getattr(session, "conversation_id", "") or "")
        session_id = str(getattr(session, "session_id", "") or "")
        self._emit(
            "router_decision",
            session_id,
            agent_name="router",
            payload={
                "conversation_id": conversation_id,
                "route": route,
                "confidence": confidence,
                "reasons": list(reasons),
                "router_method": router_method,
                "suggested_agent": suggested_agent,
                "force_agent": force_agent,
                "has_attachments": has_attachments,
            },
        )

    def build_callbacks(
        self,
        session_or_state: Any,
        *,
        trace_name: str,
        agent_name: str = "",
        job_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        base_callbacks: Optional[List[Any]] = None,
    ) -> List[Any]:
        session_id = str(getattr(session_or_state, "session_id", "") or "")
        conversation_id = str(getattr(session_or_state, "conversation_id", "") or "")
        combined_metadata = {
            "conversation_id": conversation_id,
            **dict(metadata or {}),
        }
        callbacks = list(base_callbacks or [])
        callbacks.extend(
            get_langchain_callbacks(
                self.settings,
                session_id=session_id,
                trace_name=trace_name,
                metadata=combined_metadata,
            )
        )
        if getattr(self.settings, "runtime_events_enabled", True):
            callbacks.append(
                RuntimeTraceCallbackHandler(
                    event_sink=self.event_sink,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    trace_name=trace_name,
                    agent_name=agent_name,
                    job_id=job_id,
                    metadata=combined_metadata,
                )
            )
        return callbacks

    def process_turn(
        self,
        session: Any,
        *,
        user_text: str,
        agent_name: Optional[str] = None,
    ) -> str:
        return self.process_agent_turn(session, user_text=user_text, agent_name=agent_name)

    def process_basic_turn(
        self,
        session: Any,
        *,
        user_text: str,
        system_prompt: str,
        chat_llm: Any,
        route_metadata: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        from agentic_chatbot_next.basic_chat import run_basic_chat

        state = self.hydrate_session_state(session)
        model_messages = [message.to_langchain() for message in state.messages]
        state.metadata["route_context"] = dict(route_metadata or {})
        state.append_message("user", user_text)
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        basic_callbacks = self.build_callbacks(
            state,
            trace_name="basic_turn",
            agent_name="basic",
            metadata={"route": "BASIC", **dict(route_metadata or {})},
            base_callbacks=callbacks,
        )
        self._emit(
            "basic_turn_started",
            state.session_id,
            agent_name="basic",
            payload={
                "conversation_id": state.conversation_id,
                "user_text": user_text[:500],
                **dict(route_metadata or {}),
            },
        )
        try:
            text = run_basic_chat(
                chat_llm,
                messages=model_messages,
                user_text=user_text,
                system_prompt=system_prompt,
                callbacks=basic_callbacks,
            )
        except Exception as exc:
            self._persist_state(state)
            state.sync_to_session(session)
            self._emit(
                "basic_turn_failed",
                state.session_id,
                agent_name="basic",
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **dict(route_metadata or {}),
                },
            )
            raise
        state.append_message("assistant", text, metadata={"agent_name": "basic"})
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        self._emit(
            "basic_turn_completed",
            state.session_id,
            agent_name="basic",
            payload={
                "conversation_id": state.conversation_id,
                "assistant_message_id": state.messages[-1].message_id,
                **dict(route_metadata or {}),
            },
        )
        self._run_post_turn_memory_maintenance(state, latest_text=user_text)
        state.sync_to_session(session)
        return text

    def process_agent_turn(
        self,
        session: Any,
        *,
        user_text: str,
        callbacks: Optional[List[Any]] = None,
        agent_name: Optional[str] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        state = self.hydrate_session_state(session)
        state.metadata["route_context"] = dict(route_metadata or {})
        state.append_message("user", user_text)
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        chosen_agent = agent_name or ("coordinator" if getattr(self.settings, "enable_coordinator_mode", False) else "general")
        agent = self._resolve_agent(chosen_agent)
        state.active_agent = agent.name
        self._persist_state(state)
        runtime_callbacks = self.build_callbacks(
            state,
            trace_name="agent_turn",
            agent_name=agent.name,
            metadata={**dict(route_metadata or {}), "requested_agent": agent.name},
            base_callbacks=callbacks,
        )
        self._emit(
            "turn_accepted",
            state.session_id,
            agent_name=agent.name,
            payload={"user_text": user_text[:500]},
        )
        self._emit(
            "agent_run_started",
            state.session_id,
            agent_name=agent.name,
            payload={"mode": agent.mode},
        )
        self._emit(
            "agent_turn_started",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                "user_text": user_text[:500],
                **dict(route_metadata or {}),
            },
        )
        try:
            result = self.run_agent(agent, state, user_text=user_text, callbacks=runtime_callbacks)
        except Exception as exc:
            self._persist_state(state)
            state.sync_to_session(session)
            self._emit(
                "agent_turn_failed",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **dict(route_metadata or {}),
                },
            )
            self._emit(
                "turn_failed",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **dict(route_metadata or {}),
                },
            )
            raise
        state.messages = result.messages
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": RuntimeMessage(role="assistant", content=result.text).to_dict()},
        )
        self._emit(
            "agent_run_completed",
            state.session_id,
            agent_name=agent.name,
            payload=dict(result.metadata),
        )
        self._emit(
            "agent_turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                **dict(route_metadata or {}),
                **dict(result.metadata),
            },
        )
        self._emit(
            "turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={"assistant_message_id": state.messages[-1].message_id if state.messages else ""},
        )
        self._run_post_turn_memory_maintenance(state, latest_text=user_text)
        state.sync_to_session(session)
        return result.text

    def run_agent(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        callbacks: List[Any],
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        if agent.mode == "coordinator":
            return self._run_coordinator(agent, session_state, user_text=user_text, callbacks=callbacks)

        tool_context = ToolContext(
            settings=self.settings,
            providers=self.providers,
            stores=self.stores,
            session=session_state,
            paths=self.paths,
            callbacks=callbacks,
            transcript_store=self.transcript_store,
            job_manager=self.job_manager,
            event_sink=self.event_sink,
            kernel=self,
            active_agent=agent.name,
            active_definition=agent,
            file_memory_store=self.file_memory_store,
            metadata={
                "task_payload": dict(task_payload or {}),
                "job_id": str((task_payload or {}).get("job_id") or ""),
            },
        )
        tools = self._build_tools(agent, tool_context)
        loop_result: QueryLoopResult = self.query_loop.run(
            agent,
            session_state,
            user_text=user_text,
            tool_context=tool_context,
            tools=tools,
            task_payload=task_payload,
        )
        return AgentRunResult(
            text=loop_result.text,
            messages=list(loop_result.messages or session_state.messages),
            metadata=dict(loop_result.metadata),
        )

    def spawn_worker_from_tool(
        self,
        tool_context: ToolContext,
        *,
        prompt: str,
        agent_name: str = "utility",
        description: str = "",
        run_in_background: bool = False,
    ) -> Dict[str, Any]:
        active_definition = tool_context.active_definition
        allowed_agents = set(active_definition.allowed_worker_agents if active_definition is not None else [])
        clean_agent = (agent_name or "utility").strip()
        if clean_agent not in allowed_agents:
            return {"error": f"Agent '{clean_agent}' is not allowed.", "allowed_agents": sorted(allowed_agents)}
        worker_agent = self._resolve_agent(clean_agent)
        scoped_state = self._build_scoped_worker_state(tool_context.session, agent_name=clean_agent)
        worker_request = WorkerExecutionRequest(
            agent_name=clean_agent,
            task_id="manual",
            title=description or prompt[:80] or clean_agent,
            prompt=prompt,
            description=description or prompt[:120],
        )
        job = self.job_manager.create_job(
            agent_name=clean_agent,
            prompt=prompt,
            session_id=tool_context.session.session_id,
            description=description or prompt[:120],
            session_state=scoped_state.to_dict(),
            metadata={
                "session_state": scoped_state.to_dict(),
                "worker_request": worker_request.to_dict(),
                "route_context": dict(tool_context.session.metadata.get("route_context") or {}),
            },
        )
        if run_in_background:
            self.job_manager.start_background_job(job, self._job_runner)
            return {"job_id": job.job_id, "status": "queued", "agent_name": clean_agent, "background": True}
        result = self.job_manager.run_job_inline(job, self._job_runner)
        refreshed = self.job_manager.get_job(job.job_id) or job
        return {
            "job_id": job.job_id,
            "status": refreshed.status,
            "agent_name": clean_agent,
            "background": False,
            "result": result,
            "output_path": refreshed.output_path,
        }

    def message_worker_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str,
        message: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        del tool_context
        mailbox_message = self.job_manager.enqueue_message(job_id, message)
        if mailbox_message is None:
            return {"error": f"Job '{job_id}' was not found."}
        if resume:
            self.job_manager.continue_job(job_id, self._job_runner)
        job = self.job_manager.get_job(job_id)
        return {"job_id": job_id, "status": getattr(job, "status", "unknown"), "queued": True}

    def list_jobs_from_tool(
        self,
        tool_context: ToolContext,
        *,
        status_filter: str = "",
    ) -> List[Dict[str, Any]]:
        jobs = self.job_manager.list_jobs(session_id=tool_context.session.session_id)
        rows: List[Dict[str, Any]] = []
        for job in jobs:
            if status_filter and job.status != status_filter:
                continue
            rows.append(
                {
                    "job_id": job.job_id,
                    "agent_name": job.agent_name,
                    "status": job.status,
                    "description": job.description,
                    "result_summary": job.result_summary,
                    "output_path": job.output_path,
                }
            )
        return rows

    def stop_job_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str,
    ) -> Dict[str, Any]:
        del tool_context
        job = self.job_manager.stop_job(job_id)
        if job is None:
            return {"error": f"Job '{job_id}' was not found."}
        return {"job_id": job.job_id, "status": job.status}

    def _run_coordinator(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        callbacks: List[Any],
    ) -> AgentRunResult:
        planner_name = str(agent.metadata.get("planner_agent") or "planner")
        finalizer_name = str(agent.metadata.get("finalizer_agent") or "finalizer")
        verifier_name = str(agent.metadata.get("verifier_agent") or "verifier")
        verify_outputs = bool(agent.metadata.get("verify_outputs", False))

        planner = self._resolve_agent(planner_name)
        finalizer = self._resolve_agent(finalizer_name)

        self._emit(
            "coordinator_planning_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "planner_agent": planner.name,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        planner_result = self.run_agent(planner, session_state, user_text=user_text, callbacks=callbacks)
        planner_payload = dict(planner_result.metadata.get("planner_payload") or {})
        if not planner_payload:
            planner_payload = extract_json(planner_result.text or "") or {}
        self._emit(
            "coordinator_planning_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "task_count": len(planner_payload.get("tasks") or []),
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )

        execution_state = TaskExecutionState(
            user_request=user_text,
            planner_summary=str(planner_payload.get("summary") or ""),
            task_plan=list(planner_payload.get("tasks") or []),
        )

        task_results: List[Dict[str, Any]] = []
        while True:
            batch = select_execution_batch(execution_state.task_plan, task_results)
            if not batch:
                break
            self._emit(
                "coordinator_batch_started",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "task_ids": [str(task.get("id") or "") for task in batch],
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            task_results.extend(
                self._run_task_batch(
                    agent=agent,
                    session_state=session_state,
                    user_request=user_text,
                    callbacks=callbacks,
                    batch=batch,
                )
            )

        execution_state.task_results = task_results
        execution_state.partial_answer = self._build_partial_answer(task_results)

        finalizer_result = self.run_agent(
            finalizer,
            session_state,
            user_text=user_text,
            callbacks=callbacks,
            task_payload=execution_state.to_dict(),
        )
        final_text = finalizer_result.text
        execution_state.final_answer = final_text
        self._emit(
            "coordinator_finalizer_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "finalizer_agent": finalizer.name,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )

        verification = VerificationResult()
        verifier = self.registry.get(verifier_name) if verify_outputs else None
        if verifier is not None:
            verifier_result = self.run_agent(
                verifier,
                session_state,
                user_text=user_text,
                callbacks=callbacks,
                task_payload=execution_state.to_dict(),
            )
            verification = self._parse_verification_result(verifier_result)
            execution_state.verification = verification.to_dict()
            self._emit(
                "coordinator_verifier_completed",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "verifier_agent": verifier.name,
                    "status": verification.status,
                    "issues": verification.issues,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            if verification.status == "revise" and verification.feedback:
                execution_state.partial_answer = final_text
                finalizer_result = self.run_agent(
                    finalizer,
                    session_state,
                    user_text=user_text,
                    callbacks=callbacks,
                    task_payload=execution_state.to_dict(),
                )
                final_text = finalizer_result.text
                execution_state.final_answer = final_text

        return AgentRunResult(
            text=final_text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=final_text)],
            metadata={
                "planner_payload": planner_payload,
                "task_execution_state": execution_state.to_dict(),
                "verification": verification.to_dict(),
            },
        )

    def _build_tools(self, agent: AgentDefinition, tool_context: ToolContext) -> List[Any]:
        if tool_context.providers is None or tool_context.stores is None:
            return []
        return build_agent_tools(agent, tool_context, policy_service=self.tool_policy)

    def _build_scoped_worker_state(self, parent: SessionState, *, agent_name: str) -> SessionState:
        return SessionState(
            tenant_id=parent.tenant_id,
            user_id=parent.user_id,
            conversation_id=parent.conversation_id,
            request_id=parent.request_id,
            session_id=parent.session_id,
            uploaded_doc_ids=list(parent.uploaded_doc_ids),
            demo_mode=parent.demo_mode,
            workspace_root=parent.workspace_root,
            metadata={
                "scoped_worker": True,
                "agent_name": agent_name,
                "parent_session_id": parent.session_id,
            },
        )

    def _recent_context_summary(self, session_state: SessionState, limit: int = 4) -> str:
        rows: List[str] = []
        for message in session_state.messages[-limit:]:
            if message.role not in {"user", "assistant"} or not message.content.strip():
                continue
            rows.append(f"{message.role}: {message.content[:300]}")
        return "\n".join(rows)

    def _build_worker_request(
        self,
        *,
        task: Dict[str, Any],
        user_request: str,
        session_state: SessionState,
        artifact_refs: Optional[List[str]] = None,
    ) -> WorkerExecutionRequest:
        doc_scope = [str(item) for item in (task.get("doc_scope") or []) if str(item)]
        skill_queries = [str(item) for item in (task.get("skill_queries") or []) if str(item)]
        context_summary = self._recent_context_summary(session_state)
        parts = [
            "You are executing a scoped task delegated by a coordinator.",
            "Work only from the task brief below. Do not assume you have the full parent conversation.",
            f"ORIGINAL_USER_REQUEST:\n{user_request}",
            f"TASK_ID: {task.get('id', '')}",
            f"TASK_TITLE: {task.get('title', '')}",
            f"TASK_INPUT:\n{task.get('input', '')}",
        ]
        if doc_scope:
            parts.append("DOCUMENT_SCOPE:\n- " + "\n- ".join(doc_scope))
        if skill_queries:
            parts.append("SKILL_HINTS:\n- " + "\n- ".join(skill_queries))
        if artifact_refs:
            parts.append("AVAILABLE_ARTIFACTS:\n- " + "\n- ".join(artifact_refs))
        if context_summary:
            parts.append(f"RECENT_PARENT_CONTEXT:\n{context_summary}")
        parts.append("Return a focused result for this task only.")
        prompt = "\n\n".join(part for part in parts if part.strip())
        return WorkerExecutionRequest(
            agent_name=str(task.get("executor") or "general"),
            task_id=str(task.get("id") or ""),
            title=str(task.get("title") or ""),
            prompt=prompt,
            description=str(task.get("title") or task.get("input") or "")[:120],
            doc_scope=doc_scope,
            skill_queries=skill_queries,
            artifact_refs=list(artifact_refs or []),
            metadata={"task_spec": dict(task)},
        )

    def _run_task_batch(
        self,
        *,
        agent: AgentDefinition,
        session_state: SessionState,
        user_request: str,
        callbacks: List[Any],
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del callbacks
        artifact_refs: List[str] = []
        jobs: List[tuple[str, WorkerExecutionRequest, Any]] = []
        for task in batch:
            worker_request = self._build_worker_request(
                task=task,
                user_request=user_request,
                session_state=session_state,
                artifact_refs=artifact_refs,
            )
            if worker_request.agent_name not in set(agent.allowed_worker_agents):
                result = TaskResult(
                    task_id=worker_request.task_id,
                    title=worker_request.title,
                    executor=worker_request.agent_name,
                    status="failed",
                    output=f"Worker '{worker_request.agent_name}' is not allowed for coordinator execution.",
                    artifact_ref=str(task.get("artifact_ref") or f"task:{worker_request.task_id}"),
                    warnings=[f"Agent '{worker_request.agent_name}' is not allowed."],
                )
                artifact_refs.append(result.artifact_ref)
                jobs.append(("synthetic", worker_request, result))
                continue
            scoped_state = self._build_scoped_worker_state(session_state, agent_name=worker_request.agent_name)
            job = self.job_manager.create_job(
                agent_name=worker_request.agent_name,
                prompt=worker_request.prompt,
                session_id=session_state.session_id,
                description=worker_request.description,
                session_state=scoped_state.to_dict(),
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": worker_request.to_dict(),
                    "route_context": dict(session_state.metadata.get("route_context") or {}),
                },
            )
            jobs.append((job.job_id, worker_request, job))

        real_jobs = [job for job_id, _, job in jobs if job_id != "synthetic"]
        run_parallel = self._should_run_task_batch_in_parallel(batch=batch, real_jobs=real_jobs)
        if run_parallel:
            for job in real_jobs:
                self.job_manager.start_background_job(job, self._job_runner)
            self._wait_for_jobs([job.job_id for job in real_jobs])
        else:
            for job in real_jobs:
                self.job_manager.run_job_inline(job, self._job_runner)

        results: List[Dict[str, Any]] = []
        for job_id, worker_request, record in jobs:
            if job_id == "synthetic":
                results.append(record.to_dict())
                continue
            job = self.job_manager.get_job(job_id) or record
            result = self._build_task_result(job, worker_request)
            artifact_refs.append(result.artifact_ref)
            results.append(result.to_dict())
        return results

    def _should_run_task_batch_in_parallel(self, *, batch: List[Dict[str, Any]], real_jobs: List[Any]) -> bool:
        if len(real_jobs) <= 1:
            return False
        if not all(str(task.get("mode", "sequential")) == "parallel" for task in batch):
            return False
        # Local Ollama workers contend for the same host model server and can stall
        # long coordinator batches; prefer deterministic serial execution there.
        if self._uses_local_ollama_workers():
            return False
        # Only enable background fan-out when the runtime can positively identify
        # a non-Ollama backend; if provider detection is ambiguous, stay serial.
        return self._can_identify_non_ollama_worker_runtime()

    def _uses_local_ollama_workers(self) -> bool:
        provider_names = {
            self._normalize_provider_name(getattr(self.settings, attr, ""))
            for attr in ("llm_provider", "judge_provider")
        }
        provider_names.discard("")
        if "ollama" in provider_names:
            return True
        if self.providers is None:
            return False
        return any(
            self._model_runtime_name(getattr(self.providers, attr, None)) == "ollama"
            for attr in ("chat", "judge")
        )

    def _can_identify_non_ollama_worker_runtime(self) -> bool:
        provider_names = {
            self._normalize_provider_name(getattr(self.settings, attr, ""))
            for attr in ("llm_provider", "judge_provider")
        }
        provider_names.discard("")
        if provider_names:
            return True
        if self.providers is None:
            return False
        return any(
            bool(self._model_runtime_name(getattr(self.providers, attr, None)))
            for attr in ("chat", "judge")
        )

    @staticmethod
    def _normalize_provider_name(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _model_runtime_name(cls, model: Any) -> str:
        if model is None:
            return ""
        module_name = cls._normalize_provider_name(getattr(model.__class__, "__module__", ""))
        class_name = cls._normalize_provider_name(getattr(model.__class__, "__name__", ""))
        runtime_label = f"{module_name} {class_name}"
        if "ollama" in runtime_label:
            return "ollama"
        if "azure" in runtime_label:
            return "azure"
        if "openai" in runtime_label:
            return "openai"
        return ""

    def _wait_for_jobs(self, job_ids: List[str], *, timeout_seconds: float = 300.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        pending = set(job_ids)
        while pending:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for jobs: {sorted(pending)}")
            completed = set()
            for job_id in pending:
                job = self.job_manager.get_job(job_id)
                if job is not None and job.status in TERMINAL_TASK_STATUSES:
                    completed.add(job_id)
            pending -= completed
            if pending:
                time.sleep(0.02)

    def _build_task_result(self, job: JobRecord, worker_request: WorkerExecutionRequest) -> TaskResult:
        warnings: List[str] = []
        if job.last_error:
            warnings.append(str(job.last_error))
        output = str(job.result_summary or "")
        if job.output_path:
            try:
                output = Path(job.output_path).read_text(encoding="utf-8")
            except Exception:
                output = output or str(job.result_summary or "")
        return TaskResult(
            task_id=worker_request.task_id,
            title=worker_request.title,
            executor=worker_request.agent_name,
            status=str(job.status or "failed"),
            output=output,
            artifact_ref=job.output_path or f"task:{worker_request.task_id}",
            warnings=warnings,
        )

    def _build_partial_answer(self, task_results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for result in task_results:
            title = str(result.get("title") or result.get("task_id") or "Task")
            output = str(result.get("output") or "").strip()
            if not output:
                continue
            parts.append(f"{title}:\n{output}")
        return "\n\n".join(parts).strip()

    def _parse_verification_result(self, result: AgentRunResult) -> VerificationResult:
        payload = dict(result.metadata.get("verification") or {})
        if not payload:
            payload = extract_json(result.text or "") or {}
        status = str(payload.get("status") or "pass").strip().lower()
        if status not in {"pass", "revise"}:
            status = "pass"
        summary = str(payload.get("summary") or result.text or "").strip()
        issues = [str(item) for item in (payload.get("issues") or []) if str(item)]
        feedback = str(payload.get("feedback") or "\n".join(issues) or summary).strip()
        return VerificationResult(
            status=status,
            summary=summary,
            issues=issues,
            feedback=feedback,
        )

    def _job_runner(self, job: JobRecord) -> str:
        agent = self._resolve_agent(job.agent_name)
        session_payload = dict(job.metadata.get("session_state") or job.session_state or {})
        session_state = SessionState.from_dict(session_payload)
        session_state.metadata["route_context"] = dict(job.metadata.get("route_context") or {})
        worker_request = dict(job.metadata.get("worker_request") or {})
        mailbox = self.job_manager.drain_mailbox(job.job_id)
        prompt = "\n\n".join(item.content for item in mailbox if item.content.strip()) if mailbox else job.prompt
        callbacks = self.build_callbacks(
            session_state,
            trace_name="worker_job",
            agent_name=agent.name,
            job_id=job.job_id,
            metadata={
                **dict(session_state.metadata.get("route_context") or {}),
                "worker_request": worker_request,
            },
        )
        self._emit(
            "worker_agent_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": str(worker_request.get("task_id") or ""),
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        result = self.run_agent(
            agent,
            session_state,
            user_text=prompt,
            callbacks=callbacks,
            task_payload={
                "job_id": job.job_id,
                "worker_request": worker_request,
                "skill_queries": worker_request.get("skill_queries") or [],
            },
        )
        session_state.messages = list(result.messages)
        refreshed = self.job_manager.get_job(job.job_id) or job
        refreshed.result_summary = result.text[:2000]
        refreshed.metadata["session_state"] = session_state.to_dict()
        self.transcript_store.persist_job_state(refreshed)
        self.transcript_store.append_job_transcript(
            job.job_id,
            {"kind": "assistant", "content": result.text, "agent_name": agent.name},
        )
        self._emit(
            "worker_agent_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": str(worker_request.get("task_id") or ""),
                "output_path": refreshed.output_path,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        notification = self.job_manager.build_notification(refreshed)
        self._append_notification(notification, session_state.session_id)
        return result.text

    def _validate_registry(self) -> None:
        available_tools = set(build_tool_definitions(None).keys())
        available_agents = {definition.name for definition in self.registry.list()}
        prompt_dir = Path(getattr(self.settings, "skills_dir", Path("data") / "skills"))
        errors: List[str] = []
        for agent in self.registry.list():
            for tool_name in agent.allowed_tools:
                if tool_name not in available_tools:
                    errors.append(f"agent {agent.name!r} references unknown tool {tool_name!r}")
            for worker_name in agent.allowed_worker_agents:
                if worker_name not in available_agents:
                    errors.append(f"agent {agent.name!r} references unknown worker {worker_name!r}")
            for scope in agent.memory_scopes:
                try:
                    MemoryScope(scope)
                except ValueError:
                    errors.append(f"agent {agent.name!r} declares invalid memory scope {scope!r}")
            if agent.prompt_file and not (prompt_dir / agent.prompt_file).exists():
                errors.append(f"agent {agent.name!r} prompt file {agent.prompt_file!r} was not found")
        if errors:
            raise ValueError("Invalid next-runtime agent configuration:\n- " + "\n- ".join(errors))

    def _run_post_turn_memory_maintenance(self, session_state: SessionState, *, latest_text: str) -> None:
        if not latest_text.strip():
            return
        recent_messages = session_state.messages[-6:]
        scopes = [MemoryScope.conversation.value, MemoryScope.user.value]
        self._emit(
            "memory_extraction_started",
            session_state.session_id,
            agent_name="memory_maintainer",
            payload={
                "conversation_id": session_state.conversation_id,
                "scopes": scopes,
            },
        )
        try:
            saved = self.memory_extractor.apply_from_messages(
                session_state,
                recent_messages,
                scopes=scopes,
            )
        except Exception as exc:
            self._emit(
                "memory_extraction_failed",
                session_state.session_id,
                agent_name="memory_maintainer",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "error": str(exc)[:1000],
                    "scopes": scopes,
                },
            )
            return
        if saved:
            self._emit(
                "memory_extraction_completed",
                session_state.session_id,
                agent_name="memory_maintainer",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "saved_entries": saved,
                    "scopes": scopes,
                },
            )

    def _resolve_agent(self, agent_name: str) -> AgentDefinition:
        agent = self.registry.get(agent_name)
        if agent is None:
            raise ValueError(f"Runtime agent {agent_name!r} is not defined.")
        return agent

    def _load_or_build_session_state(self, session: Any) -> SessionState:
        incoming = SessionState.from_session(session)
        stored = self.transcript_store.load_session_state(incoming.session_id)
        state = stored or incoming
        if not state.workspace_root:
            state.workspace_root = str(self.paths.workspace_dir(state.session_id))
        state.metadata.setdefault("runtime_kind", "next")
        return state

    def _persist_state(self, state: SessionState) -> None:
        self.transcript_store.persist_session_state(state)

    def _append_notification(self, notification: TaskNotification, session_id: str) -> None:
        self.transcript_store.append_session_transcript(
            session_id,
            {"kind": "notification", "notification": notification.to_dict()},
        )
        self.notification_store.append(session_id, notification)
        self._emit(
            "notification_appended",
            session_id,
            agent_name=str(notification.metadata.get("agent_name") or ""),
            payload={"job_id": notification.job_id, "status": notification.status},
            job_id=notification.job_id,
        )

    def _drain_pending_notifications(self, session_state: SessionState) -> None:
        for notification in self.notification_store.drain(session_state.session_id):
            session_state.add_notification(notification)

    def _emit(
        self,
        event_type: str,
        session_id: str,
        *,
        agent_name: str = "",
        payload: Optional[Dict[str, Any]] = None,
        tool_name: str = "",
        job_id: str = "",
    ) -> None:
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=session_id,
                agent_name=agent_name,
                tool_name=tool_name,
                job_id=job_id,
                payload=dict(payload or {}),
            )
        )
