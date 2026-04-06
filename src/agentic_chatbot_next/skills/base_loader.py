from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_chatbot_next.config import Settings

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, str] = {
    "shared": "",
    "general_agent": (
        "You are the default session agent for the next runtime.\n\n"
        "Operating rules:\n"
        "- Solve straightforward requests directly in the current session whenever you can do so with your own tools.\n"
        "- Delegate only when the task is better handled by a scoped worker or when the user clearly needs a longer-running workflow.\n"
        "- Delegate multi-step research, comparisons, verification-heavy work, or parallel workflows to the coordinator.\n"
        "- Use rag_agent_tool for indexed documents, uploaded files, contracts, policies, requirements, procedures, or anything that needs grounded citations.\n"
        "- Present rag_agent_tool answers as user-facing prose, not raw JSON.\n"
        "- Use calculator for arithmetic, list_indexed_docs for document discovery, and file-backed memory tools only for user-confirmed facts.\n"
        "- Use search_skills when you need operating guidance for an unfamiliar case.\n"
        "- Preserve citations, warnings, and uncertainty in the final answer.\n"
    ),
    "rag_agent": (
        "You are a specialist RAG agent.\n\n"
        "Answer the user's QUERY using only retrieved evidence from the indexed documents.\n"
        "Cite inline using (citation_id) values taken from the evidence.\n"
        "Never fabricate document content or claim evidence that was not retrieved.\n"
    ),
    "supervisor_agent": (
        "You are the coordinator for the next runtime.\n"
        "Use spawn_worker, message_worker, list_jobs, and stop_job to coordinate complex work.\n"
        "Keep worker briefs self-contained, parallelize only independent work, and synthesize through the finalizer.\n"
    ),
    "utility_agent": (
        "You are a utility agent that handles calculations, document listing, and persistent memory.\n"
        "Tools: calculator, list_indexed_docs, memory_save, memory_load, memory_list.\n"
        "Always use calculator for arithmetic and use file-backed memory sparingly.\n"
    ),
    "basic_chat": (
        "You are a helpful assistant. "
        "Answer the user's question directly and concisely. "
        "If you are unsure, say so and suggest what information would help."
    ),
    "planner_agent": (
        "You are the planner for the next runtime.\n\n"
        "Break the user's request into a compact list of executable tasks.\n"
        "Each task must include: id, title, executor, mode, depends_on, input, doc_scope, skill_queries.\n"
        "Executors: rag_worker, utility, data_analyst, general.\n"
        "Use mode='parallel' only for independent tasks. Keep the plan bounded.\n"
    ),
    "finalizer_agent": (
        "You are the final response agent.\n"
        "Combine completed task results into a concise answer.\n"
        "Preserve citations, gaps, and uncertainties from executor outputs.\n"
        "Do not invent evidence that is not present in task artifacts.\n"
    ),
    "verifier_agent": (
        "You are the verification agent for the next runtime.\n"
        "Review the proposed final answer against the task execution state.\n"
        "Return JSON only with keys: status, summary, issues, feedback.\n"
        "Set status='revise' only when the answer is missing evidence, overstates confidence, or ignores failed tasks.\n"
    ),
    "data_analyst_agent": (
        "You are a data analyst agent. Analyze tabular data using Python pandas.\n\n"
        "Operating rules:\n"
        "1. Always call load_dataset first.\n"
        "2. Call inspect_columns before coding.\n"
        "3. Write a plan to the scratchpad before executing code.\n"
        "4. Use execute_code to run Python in the Docker sandbox.\n"
        "5. Verify results make sense before reporting.\n"
        "6. Summarize findings clearly in natural language.\n"
    ),
}

_REQUIRED_SECTIONS: Dict[str, list[str]] = {
    "rag_agent": ["cite"],
    "supervisor_agent": ["spawn_worker", "message_worker", "list_jobs", "stop_job"],
    "general_agent": ["Operating rules"],
    "planner_agent": ["executor"],
    "finalizer_agent": ["final"],
    "verifier_agent": ["status", "issues", "feedback"],
    "data_analyst_agent": ["Operating rules"],
}


@dataclass
class _CacheEntry:
    content: str
    mtime: float
    path: Path


class SkillsLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: Dict[str, _CacheEntry] = {}

    def load(
        self,
        agent_key: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        shared = self._load_file("shared") or ""
        specific = self._load_file(agent_key)
        body = specific if specific is not None else _DEFAULTS.get(agent_key, "")
        prompt = f"{shared}\n\n---\n\n{body}".strip() if shared else body.strip()
        if context:
            from agentic_chatbot_next.prompting import render_template

            prompt = render_template(prompt, context)
        return prompt

    def invalidate(self, agent_key: Optional[str] = None) -> None:
        if agent_key is None:
            self._cache.clear()
        else:
            self._cache.pop(agent_key, None)

    def _load_file(self, agent_key: str) -> Optional[str]:
        path = self._get_path(agent_key)
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._cache.pop(agent_key, None)
            return None
        except OSError as exc:
            logger.warning("Could not stat skill file %s: %s", path, exc)
            return None

        cached = self._cache.get(agent_key)
        if cached is not None and cached.mtime == mtime:
            return cached.content

        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read skill file %s: %s", path, exc)
            return None
        if not content:
            self._cache.pop(agent_key, None)
            return None

        self._validate_sections(agent_key, content, path)
        self._cache[agent_key] = _CacheEntry(content=content, mtime=mtime, path=path)
        return content

    def _get_path(self, agent_key: str) -> Optional[Path]:
        settings = self._settings
        skills_dir = getattr(settings, "skills_dir", None)
        if not isinstance(skills_dir, Path):
            skills_dir = None
        verifier_path = getattr(settings, "verifier_agent_skills_path", None)
        if not isinstance(verifier_path, Path):
            verifier_path = None
        mapping: Dict[str, Optional[Path]] = {
            "shared": getattr(settings, "shared_skills_path", None),
            "general_agent": getattr(settings, "general_agent_skills_path", None),
            "rag_agent": getattr(settings, "rag_agent_skills_path", None),
            "supervisor_agent": getattr(settings, "supervisor_agent_skills_path", None),
            "utility_agent": getattr(settings, "utility_agent_skills_path", None),
            "basic_chat": getattr(settings, "basic_chat_skills_path", None),
            "planner_agent": getattr(settings, "planner_agent_skills_path", None),
            "finalizer_agent": getattr(settings, "finalizer_agent_skills_path", None),
            "data_analyst_agent": getattr(settings, "data_analyst_skills_path", None),
            "verifier_agent": verifier_path or (skills_dir / "verifier_agent.md" if skills_dir is not None else None),
        }
        return mapping.get(agent_key)

    def _validate_sections(self, agent_key: str, content: str, path: Path) -> None:
        for section in _REQUIRED_SECTIONS.get(agent_key, []):
            if section.lower() not in content.lower():
                logger.warning(
                    "Skill file %s (agent=%r) is missing expected section %r.",
                    path,
                    agent_key,
                    section,
                )


_LOADER_CACHE: Dict[int, SkillsLoader] = {}


def get_skills_loader(settings: Settings) -> SkillsLoader:
    key = id(settings)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = SkillsLoader(settings)
    return _LOADER_CACHE[key]


def load_shared_skills(settings: Settings) -> str:
    loader = get_skills_loader(settings)
    return loader._load_file("shared") or ""  # noqa: SLF001


def _load(settings: Settings, key: str, *, context: Optional[Dict[str, Any]] = None) -> str:
    return get_skills_loader(settings).load(key, context=context)


def load_general_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "general_agent", context=context)


def load_rag_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "rag_agent", context=context)


def load_supervisor_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "supervisor_agent", context=context)


def load_utility_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "utility_agent", context=context)


def load_basic_chat_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "basic_chat", context=context)


def load_data_analyst_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "data_analyst_agent", context=context)


def load_planner_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "planner_agent", context=context)


def load_finalizer_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "finalizer_agent", context=context)


def load_verifier_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "verifier_agent", context=context)
