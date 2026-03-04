from __future__ import annotations

import json
import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_chatbot.agents.orchestrator import ChatbotApp
from agentic_chatbot.agents.session import ChatSession
from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.context import RequestContext, build_local_context
from agentic_chatbot.providers import ProviderDependencyError, build_providers
from agentic_chatbot.rag import ingest_paths

logger = logging.getLogger(__name__)
_runtime_init_error_logged = False


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


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _get_settings()


@lru_cache(maxsize=1)
def _get_runtime() -> Runtime:
    global _runtime_init_error_logged
    settings = get_settings()
    logger.info(
        "initializing runtime providers llm=%s judge=%s embeddings=%s",
        settings.llm_provider,
        settings.judge_provider,
        settings.embeddings_provider,
    )
    try:
        providers = build_providers(settings)
        bot = ChatbotApp.create(settings, providers)
        return Runtime(settings=settings, bot=bot)
    except ProviderDependencyError as exc:
        if not _runtime_init_error_logged:
            logger.error("runtime initialization failed due to provider dependencies: %s", exc)
            _runtime_init_error_logged = True
        raise


def get_runtime() -> Runtime:
    return _get_runtime()


def get_runtime_or_503() -> Runtime:
    try:
        return get_runtime()
    except ProviderDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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


def get_request_context(
    runtime: Runtime,
    conversation_id: Optional[str],
    request_id: Optional[str],
) -> RequestContext:
    return build_local_context(
        runtime.settings,
        conversation_id=conversation_id or runtime.settings.default_conversation_id,
        request_id=request_id or "",
    )


app = FastAPI(title="Agentic Gateway", version="1.0.0")


@app.get("/health/live")
def health_live() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready(runtime: Runtime = Depends(get_runtime_or_503)) -> Dict[str, str]:
    return {"status": "ready", "model": runtime.settings.gateway_model_id}


@app.get("/v1/models")
def list_models(
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
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
    runtime: Runtime = Depends(get_runtime_or_503),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
):
    if request.model != runtime.settings.gateway_model_id:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    ctx = get_request_context(
        runtime,
        conversation_id=x_conversation_id,
        request_id=x_request_id,
    )
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
    runtime: Runtime = Depends(get_runtime_or_503),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=x_conversation_id,
        request_id=x_request_id,
    )
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
