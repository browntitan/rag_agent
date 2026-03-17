"""Skills loader — reads Markdown prompt files at runtime.

Skills files live in data/skills/:
  skills.md              – shared preamble injected into ALL agents
  general_agent.md       – GeneralAgent-specific instructions
  rag_agent.md           – RAGAgent-specific instructions
  supervisor_agent.md    – Supervisor routing instructions
  utility_agent.md       – Utility agent instructions
  basic_chat.md          – BasicChat instructions

If a file is missing, each function falls back to a hard-coded default.
Edit the .md files to change agent behaviour without touching Python code.

Hot-reload
----------
All I/O is now delegated to :class:`~agentic_chatbot.rag.skills_loader.SkillsLoader`,
which uses mtime-based caching.  Editing a .md file takes effect on the *next*
call to any ``load_*`` function — no process restart required.

Template variables
------------------
Call sites that need runtime context injection should use the loader directly::

    from agentic_chatbot.rag.skills_loader import SkillsLoader
    loader = SkillsLoader(settings)
    prompt = loader.load("rag_agent", context={"tool_list": "search_document, ..."})

Alternatively, obtain the shared loader via :func:`get_skills_loader`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agentic_chatbot.config import Settings

# ---------------------------------------------------------------------------
# Shared per-Settings loader cache
# ---------------------------------------------------------------------------

_loader_cache: Dict[int, Any] = {}  # id(settings) -> SkillsLoader


def get_skills_loader(settings: Settings) -> Any:
    """Return a :class:`~agentic_chatbot.rag.skills_loader.SkillsLoader` for *settings*.

    Instances are cached by ``id(settings)`` so that a single ``SkillsLoader``
    (and its file-mtime cache) is shared across all callers that share the same
    ``Settings`` object.
    """
    from agentic_chatbot.rag.skills_loader import SkillsLoader  # lazy to avoid cycles

    key = id(settings)
    if key not in _loader_cache:
        _loader_cache[key] = SkillsLoader(settings)
    return _loader_cache[key]


# ---------------------------------------------------------------------------
# Public convenience functions (backward-compatible API)
# ---------------------------------------------------------------------------

def load_shared_skills(settings: Settings) -> str:
    """Load data/skills/skills.md (shared preamble). Returns empty string if absent."""
    if settings.skills_backend != "local":
        raise NotImplementedError(
            f"SKILLS_BACKEND={settings.skills_backend!r} is not implemented yet. "
            "Set SKILLS_BACKEND=local for now."
        )
    loader = get_skills_loader(settings)
    return loader._load_file("shared") or ""  # noqa: SLF001


def load_general_agent_skills(
    settings: Settings,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the GeneralAgent system prompt (shared preamble + agent-specific)."""
    return get_skills_loader(settings).load("general_agent", context=context)


def load_rag_agent_skills(
    settings: Settings,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the RAGAgent system prompt (shared preamble + agent-specific)."""
    return get_skills_loader(settings).load("rag_agent", context=context)


def load_supervisor_skills(
    settings: Settings,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the Supervisor agent system prompt (shared preamble + agent-specific)."""
    return get_skills_loader(settings).load("supervisor_agent", context=context)


def load_utility_agent_skills(
    settings: Settings,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the Utility agent system prompt (shared preamble + agent-specific)."""
    return get_skills_loader(settings).load("utility_agent", context=context)


def load_basic_chat_skills(
    settings: Settings,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the BasicChat system prompt (shared preamble + agent-specific)."""
    return get_skills_loader(settings).load("basic_chat", context=context)
