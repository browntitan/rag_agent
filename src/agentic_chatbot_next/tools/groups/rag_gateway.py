from __future__ import annotations

from typing import Any, List

from agentic_chatbot_next.tools.rag_agent_tool import make_rag_agent_tool


def build_rag_gateway_tools(ctx: Any) -> List[Any]:
    return [
        make_rag_agent_tool(
            ctx.settings,
            ctx.stores,
            providers=ctx.providers,
            session=ctx.session_handle,
        )
    ]
