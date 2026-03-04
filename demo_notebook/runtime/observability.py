from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.callbacks.base import BaseCallbackHandler


class PrintTraceCallbackHandler(BaseCallbackHandler):
    """Lightweight print-based observability for notebook demos."""

    def __init__(self, *, enabled: bool = True, prefix: str = "TRACE"):
        self.enabled = enabled
        self.prefix = prefix

    def _print(self, text: str) -> None:
        if self.enabled:
            print(f"[{self.prefix}] {text}")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any, **kwargs: Any) -> Any:
        name = serialized.get("name") or serialized.get("id") or "chat_model"
        self._print(f"LLM start: {name}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        self._print("LLM end")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        name = serialized.get("name") or "tool"
        preview = input_str if len(input_str) < 160 else input_str[:157] + "..."
        self._print(f"Tool start: {name} args={preview}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        preview = output if len(str(output)) < 200 else str(output)[:197] + "..."
        self._print(f"Tool end: output={preview}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._print(f"Chain error: {error}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        self._print(f"Tool error: {error}")


def print_router_decision(route: str, confidence: float, reasons: list[str]) -> None:
    print(f"[ROUTER] route={route} confidence={confidence:.2f} reasons={', '.join(reasons)}")


def print_graph_update(event: Dict[str, Any]) -> None:
    for node, payload in event.items():
        if not isinstance(payload, dict):
            print(f"[GRAPH] node={node} payload={payload}")
            continue

        updates = []
        if "next_agent" in payload:
            updates.append(f"next_agent={payload['next_agent']}")
        if "final_answer" in payload and payload.get("final_answer"):
            final_preview = str(payload["final_answer"])
            if len(final_preview) > 140:
                final_preview = final_preview[:137] + "..."
            updates.append(f"final_answer={json.dumps(final_preview)}")
        if "rag_tasks" in payload and payload.get("rag_tasks"):
            updates.append(f"rag_tasks={len(payload.get('rag_tasks', []))}")
        if "worker_results" in payload and payload.get("worker_results"):
            updates.append(f"worker_results={len(payload.get('worker_results', []))}")

        if not updates:
            keys = ",".join(sorted(payload.keys()))
            updates.append(f"keys={keys}")

        print(f"[GRAPH] node={node} updates={' | '.join(updates)}")
