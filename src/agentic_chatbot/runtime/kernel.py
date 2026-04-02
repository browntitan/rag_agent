from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.tools import tool

from agentic_chatbot.agents.basic_chat import run_basic_chat
from agentic_chatbot.observability import RuntimeTraceCallbackHandler, get_langchain_callbacks
from agentic_chatbot.runtime.agent_registry import RuntimeAgentRegistry
from agentic_chatbot.runtime.context import (
    AgentDefinition,
    RuntimeMessage,
    SessionState,
    TaskNotification,
    ToolContext,
    ToolSpec,
)
from agentic_chatbot.runtime.events import NullRuntimeEventSink, RuntimeEvent, RuntimeEventSink
from agentic_chatbot.runtime.job_manager import RuntimeJobManager
from agentic_chatbot.runtime.loop import HybridRuntimeLoop
from agentic_chatbot.runtime.task_plan import (
    TERMINAL_TASK_STATUSES,
    TaskExecutionState,
    TaskResult,
    VerificationResult,
    WorkerExecutionRequest,
    select_execution_batch,
)
from agentic_chatbot.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


class TranscriptRuntimeEventSink(RuntimeEventSink):
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


class HybridRuntimeKernel:
    def __init__(self, settings: Any, providers: Any, stores: Any) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self.transcript_store = RuntimeTranscriptStore(settings.runtime_dir)
        event_sink: RuntimeEventSink
        if getattr(settings, "runtime_events_enabled", True):
            event_sink = TranscriptRuntimeEventSink(self.transcript_store)
        else:
            event_sink = NullRuntimeEventSink()
        self.event_sink = event_sink
        self.job_manager = RuntimeJobManager(
            self.transcript_store,
            event_sink=event_sink,
            max_worker_concurrency=settings.max_worker_concurrency,
        )
        self.agent_registry = RuntimeAgentRegistry(
            settings.agents_dir,
            env_overrides_json=getattr(settings, "agent_definitions_json", ""),
        )
        self.loop = HybridRuntimeLoop(settings, providers, stores)

    def hydrate_session_state(self, session: Any) -> SessionState:
        state = SessionState.from_chat_session(session)
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
        state = self.hydrate_session_state(session)
        model_messages = [message.to_langchain() for message in state.messages]
        state.metadata["route_context"] = dict(route_metadata or {})
        state.append_message("user", user_text)
        self.transcript_store.persist_session_state(state)
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
            self.transcript_store.persist_session_state(state)
            state.sync_to_chat_session(session)
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
        self.transcript_store.persist_session_state(state)
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
                "assistant_message_id": state.messages[-1].metadata.get("message_id", ""),
                **dict(route_metadata or {}),
            },
        )
        state.sync_to_chat_session(session)
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
        self.transcript_store.persist_session_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        chosen_agent = agent_name or ("coordinator" if self.settings.enable_coordinator_mode else "general")
        agent = self.agent_registry.get(chosen_agent)
        if agent is None:
            raise ValueError(f"Runtime agent {chosen_agent!r} is not defined.")
        runtime_callbacks = self.build_callbacks(
            state,
            trace_name="agent_turn",
            agent_name=agent.name,
            metadata={**dict(route_metadata or {}), "requested_agent": agent.name},
            base_callbacks=callbacks,
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
            self.transcript_store.persist_session_state(state)
            state.sync_to_chat_session(session)
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
            raise
        state.messages = result.messages
        self.transcript_store.persist_session_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": RuntimeMessage(role="assistant", content=result.text).to_dict()},
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
        state.sync_to_chat_session(session)
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
            callbacks=callbacks,
            transcript_store=self.transcript_store,
            job_manager=self.job_manager,
            event_sink=self.event_sink,
            active_agent=agent.name,
            metadata={"task_payload": dict(task_payload or {})},
        )
        tools = self._build_tools(agent, tool_context)
        run_payload = self.loop.run(
            agent,
            tool_context,
            user_text=user_text,
            tools=tools,
            task_payload=task_payload,
        )
        messages = run_payload.get("messages") or list(session_state.messages)
        if agent.mode == "rag":
            messages = list(session_state.messages) + [RuntimeMessage(role="assistant", content=run_payload["text"])]
        return AgentRunResult(
            text=str(run_payload.get("text") or ""),
            messages=list(messages),
            metadata={k: v for k, v in run_payload.items() if k not in {"text", "messages"}},
        )

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

        planner = self.agent_registry.get(planner_name)
        finalizer = self.agent_registry.get(finalizer_name)
        if planner is None or finalizer is None:
            raise ValueError("Coordinator requires configured planner and finalizer agents.")

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
        verifier = self.agent_registry.get(verifier_name) if verify_outputs else None
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
        registry = self._tool_registry(tool_context, agent)
        tools: List[Any] = []
        for tool_name in agent.tool_names:
            spec = registry.get(tool_name)
            if spec is None:
                continue
            for candidate in spec.builder(tool_context):
                if getattr(candidate, "name", "") == tool_name:
                    tools.append(candidate)
        return tools

    def _tool_registry(self, tool_context: ToolContext, agent: AgentDefinition) -> Dict[str, ToolSpec]:
        from agentic_chatbot.tools import calculator, make_list_docs_tool, make_memory_tools, make_rag_agent_tool
        from agentic_chatbot.tools.data_analyst_tools import make_data_analyst_tools
        from agentic_chatbot.tools.skills_search_tool import make_skills_search_tool

        def utility_cluster(ctx: ToolContext) -> Iterable[Any]:
            items = [calculator, make_list_docs_tool(self.settings, self.stores, ctx)]
            items.extend(make_memory_tools(self.stores, ctx))
            try:
                items.append(make_skills_search_tool(self.settings, stores=self.stores, session=ctx))
            except Exception:
                pass
            return items

        def general_cluster(ctx: ToolContext) -> Iterable[Any]:
            return [
                make_rag_agent_tool(
                    self.settings,
                    self.stores,
                    llm=self.providers.chat,
                    judge_llm=self.providers.judge,
                    session=ctx,
                )
            ]

        def data_analyst_cluster(ctx: ToolContext) -> Iterable[Any]:
            return make_data_analyst_tools(self.stores, ctx, settings=self.settings)

        def orchestration_cluster(ctx: ToolContext) -> Iterable[Any]:
            if not agent.allowed_worker_agents:
                return []
            return self._orchestration_tools(ctx, agent)

        specs = {
            "calculator": ToolSpec(name="calculator", builder=utility_cluster, tags=["utility"], read_only=True),
            "list_indexed_docs": ToolSpec(name="list_indexed_docs", builder=utility_cluster, tags=["docs"], read_only=True),
            "memory_save": ToolSpec(name="memory_save", builder=utility_cluster, tags=["memory"]),
            "memory_load": ToolSpec(name="memory_load", builder=utility_cluster, tags=["memory"], read_only=True),
            "memory_list": ToolSpec(name="memory_list", builder=utility_cluster, tags=["memory"], read_only=True),
            "search_skills": ToolSpec(name="search_skills", builder=utility_cluster, tags=["skills"], read_only=True),
            "rag_agent_tool": ToolSpec(name="rag_agent_tool", builder=general_cluster, tags=["rag"]),
            "load_dataset": ToolSpec(name="load_dataset", builder=data_analyst_cluster, tags=["data"], read_only=True),
            "inspect_columns": ToolSpec(name="inspect_columns", builder=data_analyst_cluster, tags=["data"], read_only=True),
            "execute_code": ToolSpec(name="execute_code", builder=data_analyst_cluster, tags=["data"], destructive=True),
            "scratchpad_write": ToolSpec(name="scratchpad_write", builder=data_analyst_cluster, tags=["scratchpad"]),
            "scratchpad_read": ToolSpec(name="scratchpad_read", builder=data_analyst_cluster, tags=["scratchpad"], read_only=True),
            "scratchpad_list": ToolSpec(name="scratchpad_list", builder=data_analyst_cluster, tags=["scratchpad"], read_only=True),
            "workspace_write": ToolSpec(name="workspace_write", builder=data_analyst_cluster, tags=["workspace"]),
            "workspace_read": ToolSpec(name="workspace_read", builder=data_analyst_cluster, tags=["workspace"], read_only=True),
            "workspace_list": ToolSpec(name="workspace_list", builder=data_analyst_cluster, tags=["workspace"], read_only=True),
            "spawn_worker": ToolSpec(name="spawn_worker", builder=orchestration_cluster, tags=["orchestration"]),
            "message_worker": ToolSpec(name="message_worker", builder=orchestration_cluster, tags=["orchestration"]),
            "list_jobs": ToolSpec(name="list_jobs", builder=orchestration_cluster, tags=["orchestration"], read_only=True),
            "stop_job": ToolSpec(name="stop_job", builder=orchestration_cluster, tags=["orchestration"]),
        }
        return specs

    def _orchestration_tools(self, tool_context: ToolContext, agent: AgentDefinition) -> List[Any]:
        allowed_agents = set(agent.allowed_worker_agents)

        @tool
        def spawn_worker(
            prompt: str,
            agent_name: str = "utility",
            description: str = "",
            run_in_background: bool = False,
        ) -> str:
            """Spawn a scoped worker from the current runtime.

            Use this for long-running work, delegated research, or specialist execution.
            """
            clean_agent = (agent_name or "utility").strip()
            if clean_agent not in allowed_agents:
                return json.dumps({"error": f"Agent '{clean_agent}' is not allowed.", "allowed_agents": sorted(allowed_agents)})
            worker_agent = self.agent_registry.get(clean_agent)
            if worker_agent is None:
                return json.dumps({"error": f"Agent '{clean_agent}' is not defined."})
            scoped_state = self._build_scoped_worker_state(tool_context.session, agent_name=clean_agent)
            job = self.job_manager.create_job(
                agent_name=clean_agent,
                prompt=prompt,
                session_id=tool_context.session.session_id,
                description=description or prompt[:120],
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": WorkerExecutionRequest(
                        agent_name=clean_agent,
                        task_id="manual",
                        title=description or prompt[:80] or clean_agent,
                        prompt=prompt,
                        description=description or prompt[:120],
                    ).to_dict(),
                },
            )
            if run_in_background:
                self.job_manager.start_background_job(job, self._job_runner)
                return json.dumps({"job_id": job.job_id, "status": "queued", "agent_name": clean_agent, "background": True})
            result = self.job_manager.run_job_inline(job, self._job_runner)
            refreshed = self.job_manager.get_job(job.job_id) or job
            return json.dumps(
                {
                    "job_id": job.job_id,
                    "status": refreshed.status,
                    "agent_name": clean_agent,
                    "background": False,
                    "result": result,
                    "output_file": refreshed.output_file,
                },
                ensure_ascii=False,
            )

        @tool
        def message_worker(job_id: str, message: str, resume: bool = True) -> str:
            """Queue a follow-up message for an existing worker job."""
            mailbox_message = self.job_manager.enqueue_message(job_id, message)
            if mailbox_message is None:
                return json.dumps({"error": f"Job '{job_id}' was not found."})
            if resume:
                self.job_manager.continue_job(job_id, self._job_runner)
            job = self.job_manager.get_job(job_id)
            return json.dumps({"job_id": job_id, "status": getattr(job, "status", "unknown"), "queued": True})

        @tool
        def list_jobs(status_filter: str = "") -> str:
            """List durable runtime jobs for the current session."""
            jobs = self.job_manager.list_jobs(session_id=tool_context.session.session_id)
            rows = []
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
                        "output_file": job.output_file,
                    }
                )
            return json.dumps(rows, ensure_ascii=False)

        @tool
        def stop_job(job_id: str) -> str:
            """Stop a background worker job."""
            job = self.job_manager.stop_job(job_id)
            if job is None:
                return json.dumps({"error": f"Job '{job_id}' was not found."})
            return json.dumps({"job_id": job.job_id, "status": job.status})

        return [spawn_worker, message_worker, list_jobs, stop_job]

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
        jobs = []
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
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": worker_request.to_dict(),
                    "route_context": dict(session_state.metadata.get("route_context") or {}),
                },
            )
            jobs.append((job.job_id, worker_request, job))

        real_jobs = [job for job_id, _, job in jobs if job_id != "synthetic"]
        run_parallel = len(real_jobs) > 1 and all(str(task.get("mode", "sequential")) == "parallel" for task in batch)
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

    def _build_task_result(self, job: Any, worker_request: WorkerExecutionRequest) -> TaskResult:
        warnings: List[str] = []
        if getattr(job, "last_error", ""):
            warnings.append(str(job.last_error))
        output = str(getattr(job, "result_summary", "") or "")
        output_file = str(getattr(job, "output_file", "") or "")
        if output_file:
            try:
                output = self.transcript_store.artifact_path(job.job_id, Path(output_file).name).read_text(encoding="utf-8")
            except Exception:
                output = output or str(getattr(job, "result_summary", "") or "")
        return TaskResult(
            task_id=worker_request.task_id,
            title=worker_request.title,
            executor=worker_request.agent_name,
            status=str(getattr(job, "status", "failed") or "failed"),
            output=output,
            artifact_ref=output_file or f"task:{worker_request.task_id}",
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

    def _job_runner(self, job: Any) -> str:
        agent = self.agent_registry.get(job.agent_name)
        if agent is None:
            raise ValueError(f"Worker agent {job.agent_name!r} is not defined.")
        session_payload = dict(job.metadata.get("session_state") or {})
        session_state = SessionState.from_dict(session_payload)
        session_state.metadata["route_context"] = dict(job.metadata.get("route_context") or {})
        worker_request = dict(job.metadata.get("worker_request") or {})
        mailbox = self.job_manager.drain_mailbox(job.job_id)
        if mailbox:
            prompt = "\n\n".join(item.content for item in mailbox if item.content.strip())
        else:
            prompt = job.prompt
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
            task_payload={"worker_request": worker_request, "skill_queries": worker_request.get("skill_queries") or []},
        )
        session_state.messages = list(result.messages)
        output_path = self.transcript_store.artifact_path(job.job_id, "output.txt")
        output_path.write_text(result.text, encoding="utf-8")
        refreshed = self.job_manager.get_job(job.job_id) or job
        refreshed.output_file = str(output_path)
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
                "output_file": str(output_path),
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        notification = self.job_manager.build_notification(refreshed)
        self._append_notification(notification, session_state.session_id)
        return result.text

    def _append_notification(self, notification: TaskNotification, session_id: str) -> None:
        self.transcript_store.append_session_transcript(
            session_id,
            {"kind": "notification", "notification": notification.__dict__},
        )
        self.transcript_store.append_session_notification(session_id, notification)

    def _drain_pending_notifications(self, session_state: SessionState) -> None:
        path = self.transcript_store.session_dir(session_state.session_id) / "notifications.jsonl"
        if not path.exists():
            return
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        path.write_text("", encoding="utf-8")
        for row in rows:
            session_state.add_notification(TaskNotification(**row))

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
