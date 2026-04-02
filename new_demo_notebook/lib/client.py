from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class GatewayClientError(RuntimeError):
    """Raised when the backend gateway returns an error response."""


@dataclass
class GatewayChatResponse:
    text: str
    raw: Dict[str, Any]


class GatewayClient:
    def __init__(self, base_url: str, *, timeout_seconds: Optional[float] = None) -> None:
        self.base_url = base_url.rstrip("/")
        effective_timeout = timeout_seconds
        if effective_timeout is None:
            effective_timeout = float(os.getenv("NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS", "180"))
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                timeout=effective_timeout,
                connect=min(10.0, effective_timeout),
                read=effective_timeout,
                write=effective_timeout,
                pool=effective_timeout,
            ),
        )

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health_ready(self) -> Dict[str, Any]:
        response = self._client.get("/health/ready")
        response.raise_for_status()
        return dict(response.json())

    def get_model_id(self) -> str:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        payload = dict(response.json())
        data = payload.get("data") or []
        if not data:
            raise GatewayClientError("Gateway returned no models.")
        return str(data[0].get("id") or "")

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GatewayChatResponse:
        payload_metadata = dict(metadata or {})
        if force_agent:
            payload_metadata["force_agent"] = True
        response = self._client.post(
            "/v1/chat/completions",
            headers={
                "X-Conversation-ID": conversation_id,
                "X-Request-ID": request_id,
            },
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "metadata": payload_metadata,
            },
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"chat failed: {response.status_code} {response.text}")
        payload = dict(response.json())
        text = str(payload["choices"][0]["message"]["content"])
        return GatewayChatResponse(text=text, raw=payload)

    def chat_turn(
        self,
        *,
        history: List[Dict[str, Any]],
        user_text: str,
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GatewayChatResponse:
        return self.chat(
            messages=list(history) + [{"role": "user", "content": user_text}],
            conversation_id=conversation_id,
            model=model,
            force_agent=force_agent,
            request_id=request_id,
            metadata=metadata,
        )

    def ingest(
        self,
        *,
        paths: List[str],
        conversation_id: str,
        source_type: str = "upload",
        request_id: str = "",
    ) -> Dict[str, Any]:
        response = self._client.post(
            "/v1/ingest/documents",
            headers={
                "X-Conversation-ID": conversation_id,
                "X-Request-ID": request_id,
            },
            json={
                "paths": paths,
                "source_type": source_type,
                "conversation_id": conversation_id,
            },
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"ingest failed: {response.status_code} {response.text}")
        return dict(response.json())
