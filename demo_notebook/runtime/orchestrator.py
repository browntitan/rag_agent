from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import NotebookSettings
from .general_agent import GENERAL_SYSTEM_PROMPT, run_general_agent
from .graph_builder import (
    SYNTHESIS_SYSTEM_PROMPT,
    UTILITY_PROMPT,
    build_initial_state,
    build_multi_agent_graph,
)
from .ingest import index_kb_corpus
from .observability import print_graph_update, print_router_decision
from .providers import ProviderBundle
from .rag_agent import RAG_SYSTEM_PROMPT, build_rag_answer_callable
from .router import route_message
from .skills import SkillProfile, build_skill_profile
from .stores import PostgresVectorStore
from .supervisor import SUPERVISOR_PROMPT
from .tools import make_calculator_tool, make_list_docs_tool, make_rag_agent_tool, make_rag_tools


@dataclass
class DemoTurnResult:
    route: str
    answer: str
    used_fallback: bool = False


class DemoOrchestrator:
    """Standalone orchestrator for the notebook deliverable."""

    def __init__(
        self,
        settings: NotebookSettings,
        providers: ProviderBundle,
        store: PostgresVectorStore,
        *,
        callback_handler: Optional[Any] = None,
    ):
        self.settings = settings
        self.providers = providers
        self.store = store
        self.callback_handler = callback_handler
        self.messages: List[Any] = []
        self._skill_profile = self._build_skill_profile()
        self._prompts = self._skill_profile.prompts

        self._utility_tools = [
            make_calculator_tool(),
            make_list_docs_tool(self.store),
        ]
        self._rag_tools = make_rag_tools(self.store, self.settings)

        rag_answer = build_rag_answer_callable(
            self.settings,
            self.providers.chat,
            self._rag_tools,
            system_prompt=self._prompts["rag"],
            callbacks=self._callbacks,
        )

        # General agent demo intentionally excludes memory tools.
        self._general_tools = [
            make_calculator_tool(),
            make_list_docs_tool(self.store),
            make_rag_agent_tool(rag_answer),
        ]

        if self._skill_profile.enabled:
            print("[NOTEBOOK] skills showcase mode active.")
            if self._skill_profile.active_files:
                for path in self._skill_profile.active_files:
                    print(f"[NOTEBOOK] skill file loaded: {path}")
            else:
                print("[NOTEBOOK] no skill files found; using base prompts.")

    @property
    def _callbacks(self) -> List[Any]:
        return [self.callback_handler] if self.callback_handler is not None else []

    def _build_skill_profile(self) -> SkillProfile:
        base_prompts = {
            "supervisor": SUPERVISOR_PROMPT,
            "rag": RAG_SYSTEM_PROMPT,
            "general": GENERAL_SYSTEM_PROMPT,
            "utility": UTILITY_PROMPT,
            "synthesis": SYNTHESIS_SYSTEM_PROMPT,
        }
        return build_skill_profile(self.settings, base_prompts)

    @property
    def active_skill_files(self) -> List[str]:
        return list(self._skill_profile.active_files)

    @property
    def skills_mode_enabled(self) -> bool:
        return bool(self._skill_profile.enabled)

    def bootstrap_kb(self, *, reindex: bool = False) -> Dict[str, int]:
        self.store.ensure_schema()
        return index_kb_corpus(self.settings, self.store, reindex=reindex)

    def run_basic(self, user_text: str) -> str:
        resp = self.providers.chat.invoke(
            [
                SystemMessage(content="You are a concise assistant."),
                *self.messages,
                HumanMessage(content=user_text),
            ],
            config={"callbacks": self._callbacks, "tags": ["demo_basic"]},
        )
        text = getattr(resp, "content", None) or str(resp)
        self.messages.append(HumanMessage(content=user_text))
        self.messages.append(AIMessage(content=text))
        return str(text)

    def run_general_agent_direct(self, user_text: str) -> str:
        answer, new_messages, stats = run_general_agent(
            self.providers.chat,
            tools=self._general_tools,
            user_text=user_text,
            system_prompt=self._prompts["general"],
            history=self.messages,
            callbacks=self._callbacks,
            max_steps=self.settings.max_agent_steps,
            max_tool_calls=self.settings.max_tool_calls,
        )
        self.messages = new_messages
        print(f"[GENERAL_AGENT] steps={stats['steps']} tool_calls={stats['tool_calls']}")
        return answer

    def _run_graph(self, user_text: str, *, stream_updates: bool = False) -> str:
        docs = self.store.list_documents()
        all_doc_ids = [d.doc_id for d in docs]
        graph = build_multi_agent_graph(
            chat_llm=self.providers.chat,
            utility_tools=self._utility_tools,
            rag_tools=self._rag_tools,
            all_doc_ids=all_doc_ids,
            supervisor_prompt=self._prompts["supervisor"],
            rag_system_prompt=self._prompts["rag"],
            utility_prompt=self._prompts["utility"],
            synthesis_system_prompt=self._prompts["synthesis"],
            callbacks=self._callbacks,
            max_loops=5,
            max_agent_steps=self.settings.max_agent_steps,
            max_tool_calls=self.settings.max_tool_calls,
        )

        state = build_initial_state(self.messages, user_text)

        if stream_updates:
            final_state = None
            for event in graph.stream(
                state,
                config={"callbacks": self._callbacks, "recursion_limit": 60},
                stream_mode="updates",
            ):
                print_graph_update(event)
                # Keep last observable update snapshot; full terminal state is produced by invoke below.
            final_state = graph.invoke(
                state,
                config={"callbacks": self._callbacks, "recursion_limit": 60},
            )
        else:
            final_state = graph.invoke(
                state,
                config={"callbacks": self._callbacks, "recursion_limit": 60},
            )

        self.messages = list(final_state.get("messages", self.messages))
        answer = str(final_state.get("final_answer") or "")
        if not answer:
            for m in reversed(self.messages):
                if isinstance(m, AIMessage) and getattr(m, "content", None):
                    answer = str(m.content)
                    break
        return answer or "I could not produce an answer."

    def process_turn(self, user_text: str, *, force_agent: bool = False, stream_updates: bool = False) -> DemoTurnResult:
        decision = route_message(user_text, force_agent=force_agent)
        print_router_decision(decision.route, decision.confidence, decision.reasons)

        if decision.route == "BASIC":
            answer = self.run_basic(user_text)
            return DemoTurnResult(route="BASIC", answer=answer, used_fallback=False)

        try:
            answer = self._run_graph(user_text, stream_updates=stream_updates)
            return DemoTurnResult(route="AGENT", answer=answer, used_fallback=False)
        except Exception as exc:
            print(f"[ORCHESTRATOR] Graph path failed, falling back to GeneralAgent: {exc}")
            answer = self.run_general_agent_direct(user_text)
            return DemoTurnResult(route="AGENT", answer=answer, used_fallback=True)
