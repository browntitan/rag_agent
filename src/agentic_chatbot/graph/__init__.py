"""Multi-agent graph package.

Exports the graph builder used by the orchestrator to run the
supervisor + specialist agents architecture.
"""
from __future__ import annotations

from agentic_chatbot.graph.builder import build_multi_agent_graph

__all__ = ["build_multi_agent_graph"]
