"""SkillsLoader — mtime-cached, template-aware loader for agent skill files.

Replaces the flat function-per-agent approach in skills.py with a class that:
  - Caches file content keyed by (agent_key, mtime) — hot-reloads on file change
    without a process restart.
  - Validates that required sections are present in each loaded file (warns, never raises).
  - Supports {{variable}} template substitution at load time via render_template().

Usage::

    loader = SkillsLoader(settings)

    # Basic load (shared + agent-specific, with fallback to defaults):
    prompt = loader.load("rag_agent")

    # With runtime context injection:
    prompt = loader.load(
        "rag_agent",
        context={"tool_list": "search_document, extract_clauses", "uploaded_docs": "contract.pdf"},
    )

    # Force cache eviction (e.g. after editing files in tests):
    loader.invalidate()                 # clear all
    loader.invalidate("rag_agent")      # clear one agent's entry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_chatbot.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — mirrors the constants in skills.py; kept here so SkillsLoader
# is self-contained and skills.py wrappers can delegate entirely to it.
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, str] = {
    "shared": "",
    "general_agent": (
        "You are the default session agent for the hybrid runtime.\n\n"
        "Operating rules:\n"
        "- Solve straightforward requests directly in the current session whenever you can do so with your own tools.\n"
        "- Delegate only when the task is better handled by a scoped worker or when the user clearly needs a longer-running workflow.\n"
        "- For multi-step research, comparisons across multiple sources, background work, or specialist execution you cannot do directly, delegate to the coordinator with spawn_worker.\n"
        "- Do not orchestrate multiple workers yourself. If the task needs planning, batching, or synthesis across workers, hand it to the coordinator.\n"
        "- Use rag_agent_tool for indexed documents, uploaded files, contracts, policies, requirements, procedures, or anything that needs grounded citations.\n"
        "- Present rag_agent_tool answers as user-facing prose, not raw JSON.\n"
        "- Use calculator for arithmetic, list_indexed_docs for document discovery, and memory tools for persistent user-confirmed facts.\n"
        "- Use search_skills when you need operating guidance for an unfamiliar case.\n"
        "- Preserve citations, warnings, and uncertainty in the final answer.\n"
    ),
    "rag_agent": (
        "You are a specialist RAG (Retrieval-Augmented Generation) agent.\n\n"
        "Your job is to answer the user's QUERY using ONLY evidence retrieved from the indexed documents.\n\n"
        "Operating rules:\n"
        "1. ALWAYS call resolve_document first if the user refers to a document by name or description.\n"
        "2. Call list_document_structure to understand a document's clause/section outline before "
        "extracting specific clauses.\n"
        "3. Use search_document for targeted single-doc search; search_all_documents for broad search.\n"
        "4. Use extract_clauses when the user references specific numbered clauses (e.g. 'Clause 33').\n"
        "5. Use extract_requirements when asked to 'find all requirements', 'list all shall statements', "
        "or similar extraction tasks.\n"
        "6. For cross-document comparisons:\n"
        "   - Use diff_documents to get the structural outline diff first.\n"
        "   - Then use compare_clauses for specific clause-by-clause analysis.\n"
        "   - Store partial results with scratchpad_write between steps.\n"
        "7. When processing multiple documents sequentially (e.g. 'look at doc_1 then doc_2'), "
        "resolve each document separately, process them in order, and use scratchpad_write to "
        "accumulate findings before synthesising a final answer.\n"
        "8. If evidence is insufficient after multiple searches, clearly state what was not found.\n"
        "9. NEVER fabricate document content — only report what the tools return.\n"
        "10. Cite inline using (chunk_id) values from the tool results.\n"
    ),
    "supervisor_agent": (
        "You are the coordinator for the hybrid runtime.\n"
        "Use spawn_worker, message_worker, list_jobs, and stop_job to coordinate complex work.\n"
        "Keep worker briefs self-contained, parallelize only truly independent work, collect task outputs, and use verification when the answer is high-stakes or citation-sensitive.\n"
    ),
    "utility_agent": (
        "You are a utility agent that handles calculations, document listing, and persistent memory.\n"
        "Tools: calculator, list_indexed_docs, memory_save, memory_load, memory_list.\n"
        "Always use the calculator for math. Always call memory_load to recall facts.\n"
    ),
    "basic_chat": (
        "You are a helpful assistant. "
        "Answer the user's question directly and concisely. "
        "If you are unsure, say so and suggest what information would help."
    ),
    "planner_agent": (
        "You are the planner for the hybrid runtime.\n\n"
        "Break the user's request into a compact list of executable tasks.\n"
        "Each task must include: id, title, executor, mode, depends_on, input, doc_scope, skill_queries.\n"
        "Executors: rag_worker, utility, data_analyst, general.\n"
        "The coordinator will execute these tasks as scoped worker jobs and pass the results to a finalizer.\n"
        "Use mode='parallel' only for independent tasks. Keep the plan bounded.\n"
    ),
    "finalizer_agent": (
        "You are the final response agent.\n"
        "Combine completed task results into a concise answer.\n"
        "Preserve citations, gaps, and uncertainties from executor outputs.\n"
        "Incorporate verification feedback when present.\n"
        "Do not invent evidence that is not present in task artifacts.\n"
    ),
    "verifier_agent": (
        "You are the verification agent for the hybrid runtime.\n"
        "Review the proposed final answer against the task execution state.\n"
        "Return JSON only with keys: status, summary, issues, feedback.\n"
        "Set status='revise' only when the answer is missing evidence, overstates confidence, or ignores failed tasks.\n"
    ),
    "data_analyst_agent": (
        "You are a data analyst agent. Analyze tabular data (Excel, CSV) using Python pandas.\n\n"
        "Operating rules:\n"
        "1. ALWAYS call load_dataset first to understand the data structure.\n"
        "2. Call inspect_columns to understand distributions and nulls before coding.\n"
        "3. Write a plan to the scratchpad before executing code.\n"
        "4. Use execute_code to run Python in a secure Docker sandbox.\n"
        "5. Verify results make sense before reporting.\n"
        "6. Summarize findings clearly in natural language.\n"
    ),
}

# Sections that MUST appear (case-insensitive) in a valid skills file.
# Missing sections emit a warning — they never block execution.
_REQUIRED_SECTIONS: Dict[str, list] = {
    "rag_agent":          ["Operating rules"],
    "supervisor_agent":   ["spawn_worker", "message_worker", "list_jobs", "stop_job"],
    "general_agent":      ["Operating rules"],
    "planner_agent":      ["executor"],
    "finalizer_agent":    ["final"],
    "verifier_agent":     ["status", "issues", "feedback"],
    "data_analyst_agent": ["Operating Rules"],
}


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    content: str
    mtime: float
    path: Path


# ---------------------------------------------------------------------------
# SkillsLoader
# ---------------------------------------------------------------------------

class SkillsLoader:
    """Mtime-cached, template-aware loader for agent skill prompt files.

    One instance per ``Settings`` object is recommended (see ``_get_loader()``
    in ``skills.py``). The cache is process-local and not thread-safe; however
    in practice the chatbot runs single-threaded graph invocations.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: Dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        agent_key: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the fully assembled skill prompt for *agent_key*.

        The returned string is::

            {shared preamble}

            ---

            {agent-specific instructions}

        If the corresponding .md files are missing, built-in defaults are used.
        If *context* is provided, ``{{variable}}`` placeholders are substituted
        before returning.

        Args:
            agent_key: One of "shared", "general_agent", "rag_agent",
                "supervisor_agent", "utility_agent", "basic_chat",
                "planner_agent", "finalizer_agent", "verifier_agent", "data_analyst_agent".
            context:   Optional dict of template variables, e.g.
                ``{"tool_list": "search_document, ...", "tenant_name": "ACME"}``.
        """
        shared = self._load_file("shared") or ""
        specific = self._load_file(agent_key)
        body = specific if specific is not None else _DEFAULTS.get(agent_key, "")

        if shared:
            full = f"{shared}\n\n---\n\n{body}".strip()
        else:
            full = body.strip()

        if context:
            from agentic_chatbot.prompting import render_template
            full = render_template(full, context)

        return full

    def invalidate(self, agent_key: Optional[str] = None) -> None:
        """Evict cached entries.

        Args:
            agent_key: Specific key to evict, or ``None`` to clear everything.
        """
        if agent_key is None:
            self._cache.clear()
        else:
            self._cache.pop(agent_key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, agent_key: str) -> Optional[str]:
        """Return the raw content of the .md file for *agent_key*, or None.

        Uses mtime-based caching: the file is re-read only when its modification
        time has changed since the last load.
        """
        path = self._get_path(agent_key)
        if path is None:
            return None

        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._cache.pop(agent_key, None)
            return None
        except OSError as exc:
            logger.warning("Skills file %s stat failed: %s", path, exc)
            return None

        cached = self._cache.get(agent_key)
        if cached is not None and cached.mtime == mtime:
            return cached.content  # cache hit — file unchanged

        # Cache miss or stale — reload from disk
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Skills file %s read failed: %s", path, exc)
            return None

        if not content:
            self._cache.pop(agent_key, None)
            return None  # treat empty file as "use default"

        self._validate_sections(agent_key, content, path)
        self._cache[agent_key] = _CacheEntry(content=content, mtime=mtime, path=path)
        return content

    def _get_path(self, agent_key: str) -> Optional[Path]:
        s = self._settings
        verifier_path = getattr(s, "verifier_agent_skills_path", None)
        if not isinstance(verifier_path, Path):
            verifier_path = None
        skills_dir = getattr(s, "skills_dir", None)
        if not isinstance(skills_dir, Path):
            skills_dir = None
        mapping: Dict[str, Optional[Path]] = {
            "shared":             getattr(s, "shared_skills_path", None),
            "general_agent":      getattr(s, "general_agent_skills_path", None),
            "rag_agent":          getattr(s, "rag_agent_skills_path", None),
            "supervisor_agent":   getattr(s, "supervisor_agent_skills_path", None),
            "utility_agent":      getattr(s, "utility_agent_skills_path", None),
            "basic_chat":         getattr(s, "basic_chat_skills_path", None),
            "planner_agent":      getattr(s, "planner_agent_skills_path", None),
            "finalizer_agent":    getattr(s, "finalizer_agent_skills_path", None),
            "data_analyst_agent": getattr(s, "data_analyst_skills_path", None),
            "verifier_agent":     verifier_path or (skills_dir / "verifier_agent.md" if skills_dir is not None else None),
        }
        return mapping.get(agent_key)

    def _validate_sections(self, agent_key: str, content: str, path: Path) -> None:
        """Emit warnings if required section markers are absent (never raises)."""
        required = _REQUIRED_SECTIONS.get(agent_key, [])
        lower = content.lower()
        for section in required:
            if section.lower() not in lower:
                logger.warning(
                    "Skills file %s (agent=%r) is missing expected section %r. "
                    "The agent may behave incorrectly.",
                    path,
                    agent_key,
                    section,
                )
