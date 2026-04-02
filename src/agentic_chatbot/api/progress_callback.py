"""LangChain callback handler that captures agent progress events to a queue.

Used by the streaming endpoint to emit real-time SSE progress events while
process_turn() is running in a background thread.
"""
from __future__ import annotations

import json
import logging
import queue
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# LangGraph node names we surface to the user.
_GRAPH_NODES = {
    "supervisor", "rag_agent", "utility_agent", "parallel_planner",
    "rag_worker", "rag_synthesizer", "data_analyst", "clarify", "evaluator",
}

# Human-readable labels for nodes
_NODE_LABELS = {
    "supervisor": "Routing request",
    "rag_agent": "Searching documents",
    "utility_agent": "Running utility tools",
    "parallel_planner": "Planning parallel search",
    "rag_worker": "Searching (parallel worker)",
    "rag_synthesizer": "Synthesizing results",
    "data_analyst": "Analyzing data",
    "clarify": "Requesting clarification",
    "evaluator": "Evaluating answer quality",
}


class ProgressCallback(BaseCallbackHandler):
    """Pushes typed progress events to a queue.Queue for SSE emission.

    Thread-safe: process_turn() runs in a background thread; the SSE
    generator reads from self.events on the main thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: queue.Queue = queue.Queue()
        self._start_times: Dict[str, float] = {}
        self._active_tool_names: Dict[str, str] = {}

    def _push(self, event: Dict[str, Any]) -> None:
        self.events.put(event)

    def mark_done(self) -> None:
        """Push the sentinel that tells the SSE generator to stop reading."""
        self.events.put(None)

    # ── LangGraph chain (node) callbacks ──────────────────────────────────

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "")
        if name not in _GRAPH_NODES:
            return
        self._start_times[str(run_id)] = time.time()
        self._push({
            "type": "agent_start",
            "node": name,
            "label": _NODE_LABELS.get(name, name),
            "timestamp": int(time.time() * 1000),
        })

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # We don't push chain end events — they clutter the UI
        self._start_times.pop(str(run_id), None)

    # ── Tool callbacks ────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "unknown_tool")
        self._start_times[str(run_id)] = time.time()
        self._active_tool_names[str(run_id)] = name
        # Parse input as JSON if possible for nicer display
        try:
            parsed_input: Any = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            parsed_input = input_str
        self._push({
            "type": "tool_call",
            "id": str(run_id),
            "tool": name,
            "input": parsed_input,
            "timestamp": int(time.time() * 1000),
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        start = self._start_times.pop(run_key, None)
        duration_ms = int((time.time() - start) * 1000) if start else None
        tool_name = self._active_tool_names.pop(run_key, "tool")
        output_str = str(output) if output is not None else ""
        # Truncate large outputs — the full output is in the final answer
        display_output = output_str[:800] + "…" if len(output_str) > 800 else output_str
        self._push({
            "type": "tool_result",
            "id": run_key,
            "tool": tool_name,
            "output": display_output,
            "duration_ms": duration_ms,
            "timestamp": int(time.time() * 1000),
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        start = self._start_times.pop(run_key, None)
        duration_ms = int((time.time() - start) * 1000) if start else None
        tool_name = self._active_tool_names.pop(run_key, "tool")
        self._push({
            "type": "tool_error",
            "id": run_key,
            "tool": tool_name,
            "error": str(error)[:200],
            "duration_ms": duration_ms,
            "timestamp": int(time.time() * 1000),
        })

    # ── LLM callbacks ─────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Don't push — too noisy; multiple LLM calls happen per node
        pass

    def raise_error(self) -> bool:
        return False  # Don't suppress exceptions
