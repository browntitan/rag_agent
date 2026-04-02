"""Skills search tool — lets agents look up relevant skill instructions at runtime.

Agents call ``search_skills(query)`` when they encounter an unfamiliar situation or
want to look up the recommended procedure for handling a specific case.  The tool
parses all skill ``.md`` files into named sections and scores them against the query
using token overlap.  No embeddings are needed — the total skills corpus is small
(~500 lines).

Example usage by an agent:
    search_skills("how to handle low confidence resolve_document")
    search_skills("multi-sheet Excel inspection", agent_filter="data_analyst_agent")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.skill_index import SkillContextResolver

logger = logging.getLogger(__name__)

# Skills file keys recognised by SkillsLoader
_ALL_SKILL_KEYS = [
    "rag_agent",
    "utility_agent",
    "data_analyst_agent",
    "supervisor_agent",
    "general_agent",
    "planner_agent",
    "finalizer_agent",
    "verifier_agent",
    "basic_chat",
    "shared",
]

# Headings that carry no useful content on their own — skip as section titles
_SKIP_HEADINGS = {
    "tools",
    "tool",
    "tools available",
    "tools section",
    "rules",
    "overview",
}


@dataclass
class _Section:
    """A parsed skill section."""

    agent_key: str
    heading: str
    content: str
    tokens: set = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.agent_key} > {self.heading}"


def _tokenise(text: str) -> set:
    """Lowercase word tokens, stripping punctuation and stop-words."""
    _STOP = {
        "a", "an", "the", "to", "of", "in", "is", "it", "for", "and", "or",
        "with", "on", "be", "use", "if", "not", "that", "this", "as", "by",
        "at", "all", "any", "when", "how", "what", "which", "do", "should",
    }
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _parse_sections(agent_key: str, content: str) -> List[_Section]:
    """Split a skills file into sections by markdown headings (##, ###)."""
    sections: List[_Section] = []
    current_heading = agent_key  # fallback if no heading found
    current_lines: List[str] = []

    for line in content.splitlines():
        heading_match = re.match(r"^#{2,3}\s+(.+)", line)
        if heading_match:
            # Flush current section
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body and current_heading.lower() not in _SKIP_HEADINGS:
                    sec = _Section(
                        agent_key=agent_key,
                        heading=current_heading,
                        content=body,
                    )
                    sec.tokens = _tokenise(current_heading + " " + body)
                    sections.append(sec)
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush final section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body and current_heading.lower() not in _SKIP_HEADINGS:
            sec = _Section(
                agent_key=agent_key,
                heading=current_heading,
                content=body,
            )
            sec.tokens = _tokenise(current_heading + " " + body)
            sections.append(sec)

    return sections


def _score_section(section: _Section, query_tokens: set) -> float:
    """Score a section against query tokens.

    Title token matches are weighted 3×; content matches 1×.
    """
    title_tokens = _tokenise(section.heading)
    title_hits = len(query_tokens & title_tokens)
    content_hits = len(query_tokens & (section.tokens - title_tokens))
    return title_hits * 3.0 + content_hits * 1.0


def _build_index(settings: Settings) -> List[_Section]:
    """Parse all skills files and return a flat list of sections."""
    from agentic_chatbot.rag.skills import get_skills_loader  # noqa: PLC0415

    loader = get_skills_loader(settings)
    sections: List[_Section] = []

    for key in _ALL_SKILL_KEYS:
        try:
            raw = loader.load(key)
        except Exception as e:
            logger.debug("Could not load skill '%s': %s", key, e)
            continue
        if not raw or not raw.strip():
            continue
        parsed = _parse_sections(key, raw)
        sections.extend(parsed)
        logger.debug("Skills index: %d sections from '%s'", len(parsed), key)

    logger.debug("Skills index built: %d total sections", len(sections))
    return sections


def _agent_scope_from_filter(agent_filter: str) -> str:
    mapping = {
        "rag_agent": "rag",
        "utility_agent": "utility",
        "data_analyst_agent": "data_analyst",
        "general_agent": "general",
        "planner_agent": "planner",
        "finalizer_agent": "finalizer",
        "verifier_agent": "verifier",
    }
    return mapping.get(agent_filter.strip(), agent_filter.strip())


def make_skills_search_tool(settings: Settings, *, stores: Any | None = None, session: Any | None = None):
    """Create a ``search_skills`` tool bound to the given settings.

    The skills index is built once when the tool is created.  Because
    :class:`~agentic_chatbot.rag.skills_loader.SkillsLoader` caches files by
    mtime, hot-reloaded content is picked up automatically on the next tool
    creation (i.e. next AGENT turn).

    Returns
    -------
    BaseTool
        A LangChain tool the agent can call as ``search_skills(query=...)``.
    """
    index: List[_Section] = _build_index(settings)
    resolver = SkillContextResolver(settings, stores) if stores is not None else None

    @tool
    def search_skills(query: str, agent_filter: str = "", top_k: int = 3) -> str:
        """Search the skills library for operational guidance, procedures, and edge-case handling.

        Call this when you are uncertain how to handle a specific situation, want to
        look up the recommended procedure for an edge case, or need to understand the
        correct behaviour for an unfamiliar scenario.

        Args:
            query:        Natural language description of what you need guidance on.
                          Example: "how to handle low confidence document resolution"
                          Example: "failure recovery when search returns empty results"
                          Example: "multi-sheet Excel inspection procedure"
            agent_filter: Optional. Restrict results to a specific skill file.
                          Values: "rag_agent", "data_analyst_agent", "utility_agent",
                          "supervisor_agent", "general_agent", "planner_agent",
                          "finalizer_agent", "verifier_agent". Leave empty to search all.
            top_k:        Maximum number of matching sections to return (default 3, max 5).

        Returns:
            Formatted text with the most relevant skill sections, each prefixed by
            [agent > Section Heading].  Returns a "no results" message if nothing matches.
        """
        top_k = min(max(1, top_k), 5)

        if resolver is not None:
            try:
                context = resolver.resolve(
                    query=query,
                    tenant_id=getattr(session, "tenant_id", settings.default_tenant_id),
                    agent_scope=_agent_scope_from_filter(agent_filter),
                    top_k=top_k,
                )
                if context.matches:
                    parts = []
                    for match in context.matches[:top_k]:
                        body = match.content
                        if len(body) > 800:
                            body = body[:800] + "\n...[truncated]"
                        parts.append(f"[{match.name} | {match.skill_id} | {match.agent_scope}]\n{body}")
                    return "\n\n---\n\n".join(parts)
            except Exception as exc:
                logger.warning("DB-backed search_skills failed, falling back to lexical index: %s", exc)

        query_tokens = _tokenise(query)
        if not query_tokens:
            return "No results: query produced no searchable tokens."

        candidates = [s for s in index if not agent_filter or s.agent_key == agent_filter.strip()]
        if not candidates:
            return f"No sections found for agent_filter={agent_filter!r}."

        scored: List[Tuple[float, _Section]] = [(_score_section(s, query_tokens), s) for s in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(score, sec) for score, sec in scored[:top_k] if score > 0]

        if not top:
            query_lower = query.lower()
            fallback = [
                (1.0, s) for s in candidates
                if any(word in s.content.lower() or word in s.heading.lower() for word in query_lower.split())
            ][:top_k]
            if fallback:
                top = fallback
            else:
                return (
                    f"No matching skill sections found for query: {query!r}. "
                    f"Try broader terms or remove agent_filter."
                )

        parts: List[str] = []
        for score, sec in top:
            body = sec.content
            if len(body) > 800:
                body = body[:800] + "\n...[truncated]"
            parts.append(f"[{sec.key}]\n{body}")

        return "\n\n---\n\n".join(parts)

    return search_skills


def make_load_skill_context_tool(settings: Settings, stores: Any, *, session: Any | None = None):
    resolver = SkillContextResolver(settings, stores)

    @tool
    def load_skill_context(query: str, agent_scope: str = "rag", top_k: int = 3) -> str:
        """Load a bounded block of DB-indexed skill guidance for a task or tool workflow."""

        context = resolver.resolve(
            query=query,
            tenant_id=getattr(session, "tenant_id", settings.default_tenant_id),
            agent_scope=agent_scope.strip() or "rag",
            top_k=min(max(1, top_k), 6),
        )
        if not context.text:
            return f"No indexed skill context found for query: {query!r}"
        return context.text

    return load_skill_context
