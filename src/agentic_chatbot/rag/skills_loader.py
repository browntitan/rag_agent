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
        "You are an agentic chatbot that can call tools to solve the user's request.\n"
        "Your priorities are: (1) correct tool selection, (2) correct tool arguments, "
        "(3) clear synthesis of tool results.\n\n"
        "Operating rules:\n"
        "- When a task requires tools, use them. When it doesn't, answer directly.\n"
        "- If multiple tools are needed, create a short numbered PLAN, then execute step-by-step.\n"
        "- Use rag_agent_tool for questions about the KB, uploaded documents, policies, "
        "contracts, requirements, runbooks, or anything that needs citations.\n"
        "- The rag_agent_tool returns JSON: {answer, citations, used_citation_ids, confidence}. "
        "Present the 'answer' field to the user with citations. Do NOT dump raw JSON.\n"
        "- For multi-document tasks (compare, diff, find requirements), call rag_agent_tool with "
        "a detailed query describing the full task. Pass preferred_doc_ids_csv to constrain scope.\n"
        "- Use scratchpad_context_key to pass previous RAG findings as context into the next call.\n"
        "- Use memory_save to persist important facts the user has confirmed across sessions.\n"
        "- If tool output is insufficient, explain what is missing and ask a follow-up question.\n"
        "- Keep the final answer concise and user-friendly.\n"
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
        "You are a supervisor agent that coordinates specialist agents.\n"
        "Route to: rag_agent (document questions), utility_agent (calculations, memory, listing docs), "
        "parallel_rag (multi-document comparison), or __end__ (simple greetings / direct answers).\n\n"
        'Respond with JSON: {"reasoning": "...", "next_agent": "...", "direct_answer": "", "rag_sub_tasks": []}\n'
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
}

# Sections that MUST appear (case-insensitive) in a valid skills file.
# Missing sections emit a warning — they never block execution.
_REQUIRED_SECTIONS: Dict[str, list] = {
    "rag_agent":        ["Operating rules"],
    "supervisor_agent": ["next_agent"],
    "general_agent":    ["Operating rules"],
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
                "supervisor_agent", "utility_agent", "basic_chat".
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
        mapping: Dict[str, Optional[Path]] = {
            "shared":           getattr(s, "shared_skills_path", None),
            "general_agent":    getattr(s, "general_agent_skills_path", None),
            "rag_agent":        getattr(s, "rag_agent_skills_path", None),
            "supervisor_agent": getattr(s, "supervisor_agent_skills_path", None),
            "utility_agent":    getattr(s, "utility_agent_skills_path", None),
            "basic_chat":       getattr(s, "basic_chat_skills_path", None),
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
