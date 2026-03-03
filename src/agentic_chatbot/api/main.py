from __future__ import annotations

import json
import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_chatbot.agents.orchestrator import ChatbotApp
from agentic_chatbot.agents.session import ChatSession
from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.context import RequestContext, build_context_from_claims
from agentic_chatbot.providers import build_providers
from agentic_chatbot.rag import ingest_paths

logger = logging.getLogger(__name__)


class OpenAIMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestDocumentsRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)
    source_type: str = "upload"


class Runtime:
    def __init__(self, settings: Settings, bot: ChatbotApp):
        self.settings = settings
        self.bot = bot


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: Dict[str, List[float]] = {}

    def allow(self, key: str, *, limit: int, now_ts: float) -> bool:
        window_start = now_ts - 60.0
        with self._lock:
            arr = self._buckets.get(key, [])
            arr = [t for t in arr if t >= window_start]
            if len(arr) >= max(1, limit):
                self._buckets[key] = arr
                return False
            arr.append(now_ts)
            self._buckets[key] = arr
            return True


_rate_limiter = _RateLimiter()


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _get_settings()


@lru_cache(maxsize=1)
def _get_runtime() -> Runtime:
    settings = get_settings()
    providers = build_providers(settings)
    bot = ChatbotApp.create(settings, providers)
    return Runtime(settings=settings, bot=bot)


def get_runtime() -> Runtime:
    return _get_runtime()


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _to_langchain_history(messages: List[OpenAIMessage]) -> tuple[List[Any], str]:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(status_code=400, detail="last message must have role='user'")

    user_text = _coerce_content(last.content).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="last user message content is empty")

    history: List[Any] = []
    for m in messages[:-1]:
        content = _coerce_content(m.content)
        if not content:
            continue
        role = (m.role or "").strip().lower()
        if role == "system":
            history.append(SystemMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
        elif role == "user":
            history.append(HumanMessage(content=content))
        else:
            # Ignore unsupported roles in v1 to preserve compatibility.
            continue

    return history, user_text


def _estimate_tokens(text: str) -> int:
    # Rough heuristic for compatibility payload only.
    return max(1, len(text) // 4) if text else 0


def _build_openai_completion_payload(model: str, content: str, prompt_tokens: int) -> Dict[str, Any]:
    completion_tokens = _estimate_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chunk_text(text: str, size: int = 180) -> Iterable[str]:
    if not text:
        yield ""
        return
    for i in range(0, len(text), size):
        yield text[i:i + size]


def _stream_chat_chunks(model: str, text: str) -> Iterable[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    for part in _chunk_text(text):
        body = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    end = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _decode_bearer_token(authorization: str, settings: Settings) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")

    if not settings.jwt_secret_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT_SECRET_KEY is not configured")

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}") from exc

    if not isinstance(claims, dict):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims")

    return claims


def get_request_context(
    request: ChatCompletionsRequest,
    runtime: Runtime,
    authorization: str,
    conversation_id: Optional[str],
    request_id: Optional[str],
) -> RequestContext:
    claims = _decode_bearer_token(authorization, runtime.settings)
    try:
        return build_context_from_claims(
            runtime.settings,
            claims,
            conversation_id=conversation_id or request.user or runtime.settings.default_conversation_id,
            request_id=request_id or "",
            fallback_user_id=request.user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def _enforce_rate_limit(ctx: RequestContext, runtime: Runtime) -> None:
    key = f"{ctx.tenant_id}:{ctx.user_id}"
    allowed = _rate_limiter.allow(
        key,
        limit=runtime.settings.rate_limit_per_minute,
        now_ts=time.time(),
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry shortly.",
        )


app = FastAPI(title="Agentic Gateway", version="1.0.0")


@app.get("/health/live")
def health_live() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready(runtime: Runtime = Depends(get_runtime)) -> Dict[str, str]:
    return {"status": "ready", "model": runtime.settings.gateway_model_id}


@app.get("/v1/models")
def list_models(
    settings: Settings = Depends(get_settings),
    authorization: str = Header(..., alias="Authorization"),
) -> Dict[str, Any]:
    _decode_bearer_token(authorization, settings)
    model_id = settings.gateway_model_id
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agentic-chatbot",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionsRequest,
    runtime: Runtime = Depends(get_runtime),
    authorization: str = Header(..., alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
):
    if request.model != runtime.settings.gateway_model_id:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    ctx = get_request_context(
        request,
        runtime,
        authorization=authorization,
        conversation_id=x_conversation_id,
        request_id=x_request_id,
    )
    _enforce_rate_limit(ctx, runtime)
    logger.info(
        "chat_completions request tenant=%s user=%s conversation=%s request_id=%s model=%s stream=%s",
        ctx.tenant_id,
        ctx.user_id,
        ctx.conversation_id,
        ctx.request_id or "-",
        request.model,
        request.stream,
    )

    history, user_text = _to_langchain_history(request.messages)
    session = ChatSession.from_context(ctx, messages=history)

    force_agent = bool(request.metadata.get("force_agent", False))
    answer = runtime.bot.process_turn(session, user_text=user_text, upload_paths=[], force_agent=force_agent)

    prompt_text = "\n".join(_coerce_content(m.content) for m in request.messages)
    prompt_tokens = _estimate_tokens(prompt_text)

    if request.stream:
        return StreamingResponse(
            _stream_chat_chunks(request.model, answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    payload = _build_openai_completion_payload(request.model, answer, prompt_tokens)
    return JSONResponse(payload)


@app.post("/v1/ingest/documents")
def ingest_documents(
    request: IngestDocumentsRequest,
    runtime: Runtime = Depends(get_runtime),
    authorization: str = Header(..., alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> Dict[str, Any]:
    # Build a minimal context using default/fallback user scope for ingestion routes.
    claims = _decode_bearer_token(authorization, runtime.settings)
    try:
        ctx = build_context_from_claims(
            runtime.settings,
            claims,
            conversation_id=x_conversation_id or runtime.settings.default_conversation_id,
            request_id=x_request_id or "",
            fallback_user_id=runtime.settings.default_user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    _enforce_rate_limit(ctx, runtime)
    logger.info(
        "ingest_documents request tenant=%s user=%s conversation=%s request_id=%s source_type=%s files=%d",
        ctx.tenant_id,
        ctx.user_id,
        ctx.conversation_id,
        ctx.request_id or "-",
        request.source_type,
        len(request.paths),
    )

    paths = [Path(p).expanduser() for p in request.paths]
    missing = [str(p) for p in paths if not p.exists()]
    valid_paths = [p for p in paths if p.exists()]

    doc_ids = ingest_paths(
        runtime.settings,
        runtime.bot.ctx.stores,
        valid_paths,
        source_type=request.source_type,
        tenant_id=ctx.tenant_id,
    )

    return {
        "object": "ingest.result",
        "tenant_id": ctx.tenant_id,
        "ingested_count": len(doc_ids),
        "doc_ids": doc_ids,
        "missing_paths": missing,
    }
