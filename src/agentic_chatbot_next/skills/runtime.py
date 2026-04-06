from __future__ import annotations

import logging
from typing import Any, Dict

from agentic_chatbot_next.agents.prompt_builder import PromptBuilder
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.skills.resolver import SkillResolver

logger = logging.getLogger(__name__)


class SkillRuntime:
    def __init__(self, settings: Any, stores: Any, prompt_builder: PromptBuilder) -> None:
        self.settings = settings
        self.stores = stores
        self.prompt_builder = prompt_builder
        self.resolver = SkillResolver(settings, stores)

    def resolve_context(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        task_payload: Dict[str, Any] | None = None,
    ) -> str:
        if not agent.skill_scope:
            return ""
        payload = dict(task_payload or {})
        worker_request = dict(payload.get("worker_request") or {})
        skill_queries = [
            str(item).strip()
            for item in (payload.get("skill_queries") or [])
            if str(item).strip()
        ]
        skill_queries.extend(
            str(item).strip()
            for item in (worker_request.get("skill_queries") or [])
            if str(item).strip()
        )
        query_parts = [user_text.strip(), *skill_queries]
        try:
            resolved = self.resolver.resolve(
                query="\n".join(part for part in query_parts if part),
                tenant_id=session_state.tenant_id,
                agent_scope=agent.skill_scope,
                tool_tags=list(agent.allowed_tools),
            )
            return resolved.text
        except Exception as exc:
            logger.warning("Skill resolution failed for %s: %s", agent.name, exc)
            return ""

    def build_prompt(self, agent: AgentDefinition) -> str:
        return self.prompt_builder.load_prompt(agent.prompt_file).strip()
