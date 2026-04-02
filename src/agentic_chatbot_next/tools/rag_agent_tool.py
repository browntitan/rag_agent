from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain.tools import tool

from agentic_chatbot_next.rag.engine import run_rag_contract


def _parse_csv(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_rag_agent_tool(
    settings: object,
    stores: object,
    *,
    providers: Any,
    session: Any,
) -> Callable:
    @tool
    def rag_agent_tool(
        query: str,
        conversation_context: str = "",
        preferred_doc_ids_csv: str = "",
        must_include_uploads: bool = True,
        top_k_vector: int = 12,
        top_k_keyword: int = 12,
        max_retries: int = 2,
        scratchpad_context_key: str = "",
    ) -> Dict[str, Any]:
        """Answer questions grounded in the KB and uploaded documents."""

        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)
        if scratchpad_context_key and scratchpad_context_key in getattr(session, "scratchpad", {}):
            extra = session.scratchpad[scratchpad_context_key]
            conversation_context = f"{extra}\n\n{conversation_context}".strip()

        callbacks: List[Any] = []
        try:
            from langchain_core.runnables.config import get_config

            cfg = get_config() or {}
            callbacks = cfg.get("callbacks") or []
        except Exception:
            callbacks = []

        contract = run_rag_contract(
            settings,
            stores,
            providers=providers,
            session=session,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=must_include_uploads,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            max_retries=max_retries,
            callbacks=callbacks,
        )
        return contract.to_dict()

    return rag_agent_tool
