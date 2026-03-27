"""Agent Registry — dynamic discovery of available specialist agents.

The :class:`AgentRegistry` holds :class:`AgentSpec` descriptors for every
specialist agent the supervisor can route to.  The supervisor asks the
registry for:

- A formatted markdown block to inject into its system prompt
  (via ``{{available_agents}}`` template variable).
- The set of valid agent names for JSON response validation.

New agents are added by registering an ``AgentSpec`` — no changes to
``supervisor.py`` or ``builder.py`` routing logic are needed for the prompt;
only the graph wiring in ``builder.py`` and the routing key in
``route_from_supervisor`` need updating.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agentic_chatbot.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    """Descriptor for a specialist agent available to the supervisor.

    Attributes:
        name:         Graph node name used in routing, e.g. ``"data_analyst"``.
        display_name: Human-readable label, e.g. ``"Data Analyst Agent"``.
        description:  2-3 sentence capability summary shown in supervisor prompt.
        use_when:     List of bullet-point conditions that trigger routing here.
        skills_key:   Key used with :class:`~agentic_chatbot.rag.skills_loader.SkillsLoader`.
        enabled:      When ``False`` this agent is hidden from the supervisor.
    """

    name: str
    display_name: str
    description: str
    use_when: List[str]
    skills_key: str
    enabled: bool = True


class AgentRegistry:
    """Central registry of available specialist agents.

    The supervisor asks this registry to render its prompt section and to
    validate response JSON.  New agents are registered here so they appear
    in the supervisor's awareness without touching ``supervisor.py``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._agents: Dict[str, AgentSpec] = {}
        self._register_builtin_agents()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, spec: AgentSpec) -> None:
        """Add or replace an agent spec. Replaces silently on re-registration."""
        self._agents[spec.name] = spec

    def get(self, name: str) -> Optional[AgentSpec]:
        """Return the spec for *name*, or ``None`` if not registered."""
        return self._agents.get(name)

    def list_enabled(self) -> List[AgentSpec]:
        """Return all currently enabled agent specs in insertion order."""
        return [a for a in self._agents.values() if a.enabled]

    def valid_agent_names(self) -> set:
        """Return the set of enabled agent names plus ``'__end__'``.

        Used by the supervisor to validate its JSON routing response.
        """
        return {a.name for a in self.list_enabled()} | {"__end__"}

    def format_for_supervisor_prompt(self) -> str:
        """Render a markdown block listing all enabled agents.

        This is injected into the supervisor system prompt via the
        ``{{available_agents}}`` template variable.
        """
        lines = ["## Available Agents\n"]
        for i, agent in enumerate(self.list_enabled(), 1):
            lines.append(f"### {i}. `{agent.name}`")
            lines.append(f"{agent.description}\n")
            lines.append("Use when:")
            for bullet in agent.use_when:
                lines.append(f"- {bullet}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Built-in agent registration
    # ------------------------------------------------------------------

    def _register_builtin_agents(self) -> None:
        """Register all built-in agents. Called once in ``__init__``."""

        self.register(AgentSpec(
            name="rag_agent",
            display_name="RAG Document Agent",
            description=(
                "Searches, extracts, and reasons over documents in the knowledge base. "
                "Handles clause extraction, requirement analysis, document comparison, "
                "and citation-grounded Q&A."
            ),
            use_when=[
                "User asks about content in uploaded documents",
                "Questions needing citations or grounded answers",
                "Clause extraction, requirement search, document comparison",
                "Any policy, contract, compliance, or knowledge base question",
            ],
            skills_key="rag_agent",
        ))

        self.register(AgentSpec(
            name="utility_agent",
            display_name="Utility Agent",
            description=(
                "Handles mathematical calculations, lists available documents, "
                "and manages persistent cross-session memory (save/load/list facts)."
            ),
            use_when=[
                "Mathematical calculations or unit conversions",
                "Listing available documents in the knowledge base",
                "Saving, loading, or listing remembered facts",
            ],
            skills_key="utility_agent",
        ))

        self.register(AgentSpec(
            name="parallel_rag",
            display_name="Parallel Document Comparison",
            description=(
                "Runs multiple RAG workers in parallel to compare or simultaneously "
                "analyze multiple specific documents. Requires specifying sub-tasks "
                "with document scopes."
            ),
            use_when=[
                "User explicitly asks to compare two or more documents",
                "Side-by-side clause-by-clause analysis of multiple contracts",
                "Diffing or simultaneously analyzing multiple specific documents",
            ],
            skills_key="supervisor_agent",
        ))

        # Data analyst — only enabled when Docker is available on this host
        docker_ok = self._check_docker_available()
        if not docker_ok:
            logger.warning(
                "Docker is not available on this host — data_analyst agent will be disabled. "
                "Start Docker and restart the app to enable it."
            )
        self.register(AgentSpec(
            name="data_analyst",
            display_name="Data Analyst Agent",
            description=(
                "Analyzes tabular data (Excel, CSV) using Python pandas in a secure Docker "
                "sandbox. Can compute statistics, filter/group/pivot data, perform calculations, "
                "and generate data-driven insights. Follows a plan-verify-reflect workflow."
            ),
            use_when=[
                "User asks to analyze data in an Excel or CSV file",
                "Statistical analysis, aggregation, or data exploration requests",
                "Questions like 'what is the average...', 'group by...', 'filter rows where...'",
                "Any request involving dataframe operations or tabular data manipulation",
            ],
            skills_key="data_analyst_agent",
            enabled=docker_ok,
        ))

        # Clarification agent — always enabled; never shown in normal agent list
        # but must appear in valid_agent_names() so supervisor JSON is accepted.
        self.register(AgentSpec(
            name="clarify",
            display_name="Clarification Node",
            description=(
                "Asks the user a follow-up question when the request is too vague "
                "or ambiguous to route safely without more information. "
                "Use this instead of guessing when critical context (e.g. which "
                "document, which metric, which time period) is missing."
            ),
            use_when=[
                "The user's request refers to 'the document' or 'the file' but no document has been uploaded",
                "The request is a single ambiguous word or phrase with no context (e.g. 'summarise', 'compare')",
                "Multiple conflicting interpretations are equally plausible and the wrong choice would produce a useless answer",
                "A required parameter (e.g. clause number, date range, document name) is clearly absent",
            ],
            skills_key="supervisor_agent",
            enabled=True,
        ))

    def _check_docker_available(self) -> bool:
        """Non-blocking Docker availability check. Returns False on any error."""
        try:
            import docker  # noqa: PLC0415
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False
