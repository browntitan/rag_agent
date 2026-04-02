from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from agentic_chatbot_next.skills.indexer import SkillContextResolver as NextSkillContextResolver


@dataclass
class SkillMatch:
    skill_id: str
    name: str
    agent_scope: str
    content: str
    chunk_index: int
    score: float


@dataclass
class ResolvedSkillContext:
    text: str
    matches: List[SkillMatch] = field(default_factory=list)


class SkillResolver:
    def __init__(self, settings: Any, stores: Any) -> None:
        self._resolver = NextSkillContextResolver(settings, stores)

    def resolve(
        self,
        *,
        query: str,
        tenant_id: str,
        agent_scope: str,
        tool_tags: List[str] | None = None,
        task_tags: List[str] | None = None,
        top_k: int | None = None,
        max_chars: int | None = None,
    ) -> ResolvedSkillContext:
        result = self._resolver.resolve(
            query=query,
            tenant_id=tenant_id,
            agent_scope=agent_scope,
            tool_tags=tool_tags,
            task_tags=task_tags,
            top_k=top_k,
            max_chars=max_chars,
        )
        return ResolvedSkillContext(
            text=result.text,
            matches=[
                SkillMatch(
                    skill_id=match.skill_id,
                    name=match.name,
                    agent_scope=match.agent_scope,
                    content=match.content,
                    chunk_index=match.chunk_index,
                    score=match.score,
                )
                for match in result.matches
            ],
        )
