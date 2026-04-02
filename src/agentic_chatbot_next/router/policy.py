from __future__ import annotations

from typing import Optional

from agentic_chatbot_next.router.router import RouterDecision


def choose_agent_name(settings: object, decision: RouterDecision) -> Optional[str]:
    if bool(getattr(settings, "enable_coordinator_mode", False)):
        return "coordinator"
    suggested = str(getattr(decision, "suggested_agent", "") or "").strip().lower()
    if suggested in {"coordinator", "data_analyst"}:
        return suggested
    return None


__all__ = ["RouterDecision", "choose_agent_name"]
