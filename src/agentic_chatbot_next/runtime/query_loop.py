from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.utils.json_utils import extract_json
from agentic_chatbot_next.basic_chat import run_basic_chat
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.general_agent import run_general_agent
from agentic_chatbot_next.memory.context_builder import MemoryContextBuilder
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.rag.engine import render_rag_contract, run_rag_contract
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.task_plan import normalise_task_plan

logger = logging.getLogger(__name__)


def _recent_conversation_context(session: SessionState, limit: int = 6) -> str:
    parts: List[str] = []
    for message in session.messages[-limit:]:
        if message.role in {"user", "assistant"} and message.content.strip():
            parts.append(f"{message.role}: {message.content[:300]}")
    return "\n".join(parts)


@dataclass
class QueryLoopResult:
    text: str
    messages: List[RuntimeMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryLoop:
    def __init__(
        self,
        *,
        settings: Any | None = None,
        providers: Any | None = None,
        stores: Any | None = None,
        skill_runtime: Any | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self.skill_runtime = skill_runtime
        self._paths = RuntimePaths.from_settings(settings) if settings is not None else None
        self._memory_store = FileMemoryStore(self._paths) if self._paths is not None else None
        self._memory_context = MemoryContextBuilder(self._memory_store) if self._memory_store is not None else None
        self._memory_extractor = MemoryExtractor(self._memory_store) if self._memory_store is not None else None

    def run(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        tool_context: Any | None = None,
        tools: Optional[List[Any]] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> QueryLoopResult:
        callbacks = list(getattr(tool_context, "callbacks", []) or [])
        if agent.mode != "memory_maintainer":
            if self.providers is None or getattr(self.providers, "chat", None) is None:
                raise RuntimeError("QueryLoop requires configured providers for live execution.")

        skill_context = ""
        if tool_context is not None and self.skill_runtime is not None:
            skill_context = self.skill_runtime.resolve_context(
                agent,
                session_state,
                user_text=user_text,
                task_payload=task_payload,
            )
            tool_context.skill_context = skill_context

        if agent.mode == "basic":
            return self._run_basic(agent, session_state, user_text=user_text, skill_context=skill_context, callbacks=callbacks)
        if agent.mode == "rag":
            return self._run_rag(agent, session_state, user_text=user_text, skill_context=skill_context, callbacks=callbacks)
        if agent.mode == "memory_maintainer":
            return self._run_memory_maintainer(agent, session_state, user_text=user_text)
        if agent.mode == "planner":
            return self._run_planner(agent, session_state, user_text=user_text, skill_context=skill_context, callbacks=callbacks)
        if agent.mode == "finalizer":
            return self._run_finalizer(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                task_payload=dict(task_payload or {}),
                callbacks=callbacks,
            )
        if agent.mode == "verifier":
            return self._run_verifier(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                task_payload=dict(task_payload or {}),
                callbacks=callbacks,
            )
        return self._run_react(
            agent,
            session_state,
            user_text=user_text,
            skill_context=skill_context,
            tool_context=tool_context,
            tools=list(tools or []),
        )

    def _build_system_prompt(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        skill_context: str = "",
    ) -> str:
        prompt = ""
        if self.skill_runtime is not None:
            prompt = self.skill_runtime.build_prompt(agent, skill_context=skill_context).strip()
        if not prompt:
            prompt = f"You are the {agent.name} agent."
        memory_context = ""
        if self._memory_context is not None:
            memory_context = self._memory_context.build_for_agent(agent, session_state)
        blocks = [prompt]
        if skill_context:
            blocks.append(f"## Skill Context\n{skill_context}")
        if memory_context:
            blocks.append(f"## Memory Context\n{memory_context}")
        prompt = "\n\n".join(block for block in blocks if block.strip()).strip()
        return prompt

    def _run_basic(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(agent, session_state, skill_context=skill_context)
        text = run_basic_chat(
            self.providers.chat,
            messages=[message.to_langchain() for message in session_state.messages[:-1]],
            user_text=user_text,
            system_prompt=system_prompt,
            callbacks=callbacks,
        )
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata={"agent_name": agent.name})],
            metadata={"agent_name": agent.name},
        )

    def _run_react(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        tool_context: Any,
        tools: List[Any],
    ) -> QueryLoopResult:
        if tool_context is None:
            raise ValueError("React execution requires a tool context.")
        system_prompt = self._build_system_prompt(agent, session_state, skill_context=skill_context)
        final_text, updated_messages, run_stats = run_general_agent(
            self.providers.chat,
            tools=tools,
            messages=[message.to_langchain() for message in session_state.messages[:-1]],
            user_text=user_text,
            system_prompt=system_prompt,
            callbacks=tool_context.callbacks,
            max_steps=agent.max_steps,
            max_tool_calls=agent.max_tool_calls,
            force_plan_execute=str(agent.metadata.get("execution_strategy") or "").lower() == "plan_execute",
        )
        messages = [RuntimeMessage.from_langchain(message) for message in updated_messages]
        tool_context.refresh_from_session_handle()
        return QueryLoopResult(
            text=final_text,
            messages=messages,
            metadata={"run_stats": run_stats, "agent_name": agent.name},
        )

    def _run_rag(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
    ) -> QueryLoopResult:
        contract = run_rag_contract(
            self.settings,
            self.stores,
            providers=self.providers,
            session=session_state,
            query=user_text,
            conversation_context=_recent_conversation_context(session_state),
            preferred_doc_ids=list(session_state.uploaded_doc_ids),
            must_include_uploads=bool(session_state.uploaded_doc_ids),
            top_k_vector=self.settings.rag_top_k_vector,
            top_k_keyword=self.settings.rag_top_k_keyword,
            max_retries=self.settings.rag_max_retries,
            callbacks=callbacks,
            skill_context=skill_context,
            task_context=user_text,
        )
        text = render_rag_contract(contract)
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata={"agent_name": agent.name})],
            metadata={"rag_contract": contract.to_dict(), "agent_name": agent.name},
        )

    def _run_memory_maintainer(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
    ) -> QueryLoopResult:
        if self._memory_extractor is None:
            raise RuntimeError("Memory maintainer requires a configured file-backed memory store.")
        scopes = list(agent.memory_scopes or ["conversation"])
        saved = self._memory_extractor.apply_from_messages(
            session_state,
            session_state.messages[-8:],
            scopes=scopes,
        )
        if not saved:
            saved = self._memory_extractor.apply_from_text(session_state, user_text, scopes=scopes)
        if saved:
            text = f"Saved {saved} memory entr{'y' if saved == 1 else 'ies'} across scopes: {', '.join(scopes)}."
        else:
            text = "No structured memory entries were detected in the request."
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata={"agent_name": agent.name})],
            metadata={"agent_name": agent.name, "saved_entries": saved},
        )

    def _run_planner(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(agent, session_state, skill_context=skill_context)
        prompt = (
            "Return JSON only with this schema:\n"
            "{"
            '"summary": "short summary",'
            '"tasks": ['
            '{"id": "task_1", "title": "...", "executor": "rag_worker|utility|data_analyst|general", '
            '"mode": "sequential|parallel", "depends_on": [], "input": "...", "doc_scope": [], "skill_queries": []}'
            "]}\n\n"
            f"Limit the number of tasks to {self.settings.planner_max_tasks}.\n"
            "Only mark tasks as parallel when they are truly independent.\n\n"
            f"USER_REQUEST:\n{user_text}"
        )
        text = ""
        try:
            response = self.providers.chat.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
                config={"callbacks": callbacks},
            )
            text = getattr(response, "content", None) or str(response)
        except Exception as exc:
            logger.warning("Planner agent failed: %s", exc)
        obj = extract_json(text or "") or {}
        task_plan = normalise_task_plan(
            obj.get("tasks"),
            query=user_text,
            max_tasks=self.settings.planner_max_tasks,
        )
        payload = {
            "summary": str(obj.get("summary") or f"Planned {len(task_plan)} task(s)."),
            "tasks": task_plan,
        }
        rendered = json.dumps(payload, ensure_ascii=False)
        return QueryLoopResult(
            text=rendered,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=rendered, metadata={"agent_name": agent.name})],
            metadata={"planner_payload": payload, "agent_name": agent.name},
        )

    def _run_finalizer(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        task_payload: Dict[str, Any],
        callbacks: List[Any],
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(agent, session_state, skill_context=skill_context)
        final_text = ""
        try:
            response = self.providers.chat.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"USER_REQUEST:\n{user_text}\n\n"
                            "TASK_EXECUTION_STATE:\n"
                            f"{json.dumps(task_payload, ensure_ascii=False, indent=2)}"
                        )
                    ),
                ],
                config={"callbacks": callbacks},
            )
            final_text = str(getattr(response, "content", None) or response).strip()
        except Exception as exc:
            logger.warning("Finalizer agent failed: %s", exc)
        if not final_text:
            final_text = str(task_payload.get("partial_answer") or "")
        return QueryLoopResult(
            text=final_text.strip(),
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=final_text.strip(), metadata={"agent_name": agent.name})],
            metadata={"task_payload": task_payload, "agent_name": agent.name},
        )

    def _run_verifier(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        task_payload: Dict[str, Any],
        callbacks: List[Any],
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(agent, session_state, skill_context=skill_context)
        verification = {
            "status": "pass",
            "summary": "No verification issues detected.",
            "issues": [],
            "feedback": "",
        }
        try:
            response = self.providers.chat.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            "Return JSON only with this schema:\n"
                            '{'
                            '"status": "pass|revise", '
                            '"summary": "short verification summary", '
                            '"issues": ["issue 1"], '
                            '"feedback": "clear revision guidance"'
                            "}\n\n"
                            f"USER_REQUEST:\n{user_text}\n\n"
                            "TASK_EXECUTION_STATE:\n"
                            f"{json.dumps(task_payload, ensure_ascii=False, indent=2)}"
                        )
                    ),
                ],
                config={"callbacks": callbacks},
            )
            text = str(getattr(response, "content", None) or response).strip()
            payload = extract_json(text or "") or {}
            status = str(payload.get("status") or "pass").strip().lower()
            if status not in {"pass", "revise"}:
                status = "pass"
            issues = [str(item) for item in (payload.get("issues") or []) if str(item).strip()]
            summary = str(payload.get("summary") or text or verification["summary"]).strip()
            feedback = str(payload.get("feedback") or "\n".join(issues) or summary).strip()
            verification = {
                "status": status,
                "summary": summary,
                "issues": issues,
                "feedback": feedback,
            }
        except Exception as exc:
            logger.warning("Verifier agent failed: %s", exc)
        rendered = json.dumps(verification, ensure_ascii=False)
        return QueryLoopResult(
            text=rendered,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=rendered, metadata={"agent_name": agent.name})],
            metadata={"verification": verification, "agent_name": agent.name},
        )
