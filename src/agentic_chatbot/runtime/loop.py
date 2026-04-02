from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot.rag.contracts import render_rag_contract
from agentic_chatbot.rag import SkillContextResolver
from agentic_chatbot.rag.agent import run_rag_agent
from agentic_chatbot.rag.skills import (
    load_data_analyst_skills,
    load_finalizer_agent_skills,
    load_general_agent_skills,
    load_planner_agent_skills,
    load_rag_agent_skills,
    load_supervisor_skills,
    load_utility_agent_skills,
    load_verifier_agent_skills,
)
from agentic_chatbot.runtime.context import AgentDefinition, RuntimeMessage, SessionState, ToolContext
from agentic_chatbot.runtime.task_plan import normalise_task_plan
from agentic_chatbot.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


def _recent_conversation_context(session: SessionState, limit: int = 6) -> str:
    parts: List[str] = []
    for message in session.messages[-limit:]:
        if message.role in {"user", "assistant"} and message.content.strip():
            parts.append(f"{message.role}: {message.content[:300]}")
    return "\n".join(parts)


class HybridRuntimeLoop:
    def __init__(self, settings: Any, providers: Any, stores: Any) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self._skill_resolver = SkillContextResolver(settings, stores)

    def run(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        *,
        user_text: str,
        tools: List[Any],
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        skill_context = tool_context.skill_context or self._resolve_skill_context(agent, tool_context, user_text)
        tool_context.skill_context = skill_context

        if agent.mode == "rag":
            return self._run_rag(agent, tool_context, user_text)
        if agent.mode == "planner":
            return self._run_planner(agent, tool_context, user_text)
        if agent.mode == "finalizer":
            return self._run_finalizer(agent, tool_context, user_text, task_payload=task_payload or {})
        if agent.mode == "verifier":
            return self._run_verifier(agent, tool_context, user_text, task_payload=task_payload or {})
        return self._run_react(agent, tool_context, user_text, tools=tools)

    def _resolve_skill_context(self, agent: AgentDefinition, tool_context: ToolContext, user_text: str) -> str:
        if not agent.skill_agent_scope:
            return ""
        task_payload = dict(tool_context.metadata.get("task_payload") or {})
        skill_queries = [
            str(item).strip()
            for item in (task_payload.get("skill_queries") or [])
            if str(item).strip()
        ]
        worker_request = dict(task_payload.get("worker_request") or {})
        skill_queries.extend(
            str(item).strip()
            for item in (worker_request.get("skill_queries") or [])
            if str(item).strip()
        )
        query_parts = [user_text.strip()]
        query_parts.extend(skill_queries)
        try:
            context = self._skill_resolver.resolve(
                query="\n".join(part for part in query_parts if part),
                tenant_id=tool_context.session.tenant_id,
                agent_scope=agent.skill_agent_scope,
                tool_tags=list(agent.tool_names),
            )
            return context.text
        except Exception as exc:
            logger.warning("Skill-context resolution failed for %s: %s", agent.name, exc)
            return ""

    def _load_prompt(self, agent: AgentDefinition) -> str:
        key = agent.prompt_key or agent.name
        mapping = {
            "general_agent": load_general_agent_skills,
            "supervisor_agent": load_supervisor_skills,
            "utility_agent": load_utility_agent_skills,
            "data_analyst_agent": load_data_analyst_skills,
            "rag_agent": load_rag_agent_skills,
            "planner_agent": load_planner_agent_skills,
            "finalizer_agent": load_finalizer_agent_skills,
            "verifier_agent": load_verifier_agent_skills,
        }
        loader = mapping.get(key, load_general_agent_skills)
        return loader(self.settings)

    def _build_system_prompt(self, agent: AgentDefinition, tool_context: ToolContext) -> str:
        prompt = self._load_prompt(agent).strip()
        if tool_context.skill_context:
            prompt = f"{prompt}\n\n## Skill Context\n{tool_context.skill_context}".strip()
        return prompt

    def _run_react(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        user_text: str,
        *,
        tools: List[Any],
    ) -> Dict[str, Any]:
        from agentic_chatbot.agents.general_agent import run_general_agent  # noqa: PLC0415

        system_prompt = self._build_system_prompt(agent, tool_context)
        final_text, updated_messages, run_stats = run_general_agent(
            self.providers.chat,
            tools=tools,
            messages=[message.to_langchain() for message in tool_context.session.messages],
            user_text=user_text,
            system_prompt=system_prompt,
            callbacks=tool_context.callbacks,
            max_steps=agent.max_steps,
            max_tool_calls=agent.max_tool_calls,
        )
        return {
            "text": final_text,
            "messages": [RuntimeMessage.from_langchain(message) for message in updated_messages],
            "run_stats": run_stats,
        }

    def _run_rag(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        user_text: str,
    ) -> Dict[str, Any]:
        contract = run_rag_agent(
            self.settings,
            self.stores,
            llm=self.providers.chat,
            judge_llm=self.providers.judge,
            query=user_text,
            conversation_context=_recent_conversation_context(tool_context.session),
            preferred_doc_ids=list(tool_context.session.uploaded_doc_ids),
            must_include_uploads=bool(tool_context.session.uploaded_doc_ids),
            top_k_vector=self.settings.rag_top_k_vector,
            top_k_keyword=self.settings.rag_top_k_keyword,
            max_retries=self.settings.rag_max_retries,
            session=tool_context,
            callbacks=tool_context.callbacks,
            skill_context=tool_context.skill_context,
            task_context=user_text,
        )
        return {
            "text": render_rag_contract(contract),
            "messages": list(tool_context.session.messages),
            "contract": contract,
        }

    def _run_planner(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        user_text: str,
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(agent, tool_context)
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
                config={"callbacks": tool_context.callbacks},
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
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "messages": list(tool_context.session.messages),
            "planner_payload": payload,
        }

    def _run_finalizer(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        user_text: str,
        *,
        task_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(agent, tool_context)
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
                config={"callbacks": tool_context.callbacks},
            )
            final_text = str(getattr(response, "content", None) or response).strip()
        except Exception as exc:
            logger.warning("Finalizer agent failed: %s", exc)
        if not final_text:
            final_text = str(task_payload.get("partial_answer") or "")
        return {
            "text": final_text.strip(),
            "messages": list(tool_context.session.messages),
            "task_payload": task_payload,
        }

    def _run_verifier(
        self,
        agent: AgentDefinition,
        tool_context: ToolContext,
        user_text: str,
        *,
        task_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(agent, tool_context)
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
                config={"callbacks": tool_context.callbacks},
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
        return {
            "text": json.dumps(verification, ensure_ascii=False),
            "messages": list(tool_context.session.messages),
            "verification": verification,
        }
