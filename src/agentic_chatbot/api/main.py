from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot.api.progress_callback import ProgressCallback

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_chatbot.agents.orchestrator import ChatbotApp
from agentic_chatbot.agents.session import ChatSession
from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.context import RequestContext, build_local_context
from agentic_chatbot.providers import (
    ProviderConfigurationError,
    ProviderDependencyError,
    build_providers,
)
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
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "When provided (or when X-Conversation-ID header is set), files are also "
            "copied into the active session workspace so the data analyst sandbox can "
            "access them at /workspace/<filename> without a separate load_dataset call."
        ),
    )


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
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        if not _runtime_init_error_logged:
            logger.error("runtime initialization failed due to provider validation: %s", exc)
            _runtime_init_error_logged = True
        raise


def get_runtime() -> Runtime:
    return _get_runtime()


def get_runtime_or_503() -> Runtime:
    try:
        return get_runtime()
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
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


def _stream_with_progress(
    model: str,
    session: Any,
    user_text: str,
    bot: Any,
    force_agent: bool,
    prompt_tokens: int,
) -> Iterable[str]:
    """SSE generator that emits real-time progress events then text content.

    Runs process_turn() in a background thread. While it runs, reads typed
    progress events from the ProgressCallback queue and yields them as named
    SSE events. After completion yields the text content as chat.completion.chunk
    events.

    Named SSE event format:
        event: progress
        data: {"type": "agent_start", "node": "rag_agent", ...}

    Content chunks use the standard OpenAI format (no event: prefix):
        data: {"choices": [{"delta": {"content": "..."}}]}
    """
    progress_cb = ProgressCallback()
    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            answer = bot.process_turn(
                session,
                user_text=user_text,
                upload_paths=[],
                force_agent=force_agent,
                extra_callbacks=[progress_cb],
            )
            result_holder["answer"] = answer
        except Exception as exc:
            exc_holder["error"] = exc
        finally:
            progress_cb.mark_done()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Stream progress events while process_turn is running
    while True:
        try:
            event = progress_cb.events.get(timeout=30)  # 30s per-event timeout
        except queue.Empty:
            # Safety valve — shouldn't happen in practice
            break
        if event is None:  # sentinel: processing complete
            break
        yield f"event: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    thread.join(timeout=5)

    # Check for errors
    if exc_holder.get("error"):
        err_text = f"Error: {str(exc_holder['error'])[:200]}"
        yield from _stream_chat_chunks(model, err_text)
        return

    answer = result_holder.get("answer", "")
    yield from _stream_chat_chunks(model, answer)


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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    prompt_text = "\n".join(_coerce_content(m.content) for m in request.messages)
    prompt_tokens = _estimate_tokens(prompt_text)

    if request.stream:
        return StreamingResponse(
            _stream_with_progress(
                request.model,
                session,
                user_text,
                runtime.bot,
                force_agent=force_agent,
                prompt_tokens=prompt_tokens,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    answer = runtime.bot.process_turn(session, user_text=user_text, upload_paths=[], force_agent=force_agent)
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

    # Copy ingested files into the active session workspace (if one exists).
    # The workspace directory is keyed by conversation_id; it exists on disk
    # only after a /v1/chat/completions turn has been processed for that session.
    # This allows the data analyst to access uploaded files at /workspace/<filename>
    # inside the Docker sandbox without needing a separate load_dataset call first.
    ws_conversation_id = (
        request.conversation_id
        or x_conversation_id
        or runtime.settings.default_conversation_id
    )
    ws_root = runtime.settings.workspace_dir / ws_conversation_id
    workspace_copies: List[str] = []
    if ws_root.is_dir():
        for p in valid_paths:
            try:
                shutil.copy2(p, ws_root / p.name)
                workspace_copies.append(p.name)
                logger.debug("ingest_documents: copied %s into workspace %s", p.name, ws_root)
            except Exception as cp_exc:
                logger.warning(
                    "ingest_documents: could not copy %s to workspace %s: %s",
                    p.name, ws_root, cp_exc,
                )

    result: Dict[str, Any] = {
        "object": "ingest.result",
        "tenant_id": ctx.tenant_id,
        "ingested_count": len(doc_ids),
        "doc_ids": doc_ids,
        "missing_paths": missing,
    }
    if workspace_copies:
        result["workspace_copies"] = workspace_copies
    return result


@app.post("/v1/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    source_type: str = "upload",
    runtime: Runtime = Depends(get_runtime_or_503),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> Dict[str, Any]:
    """Accept multipart file uploads from the frontend.

    Saves files to the uploads directory, ingests them into the KB,
    and optionally copies them into the session workspace.
    """
    ctx = get_request_context(
        runtime,
        conversation_id=x_conversation_id,
        request_id=x_request_id,
    )
    logger.info(
        "upload_files request tenant=%s conversation=%s files=%d",
        ctx.tenant_id,
        ctx.conversation_id,
        len(files),
    )

    uploads_dir = runtime.settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    errors: List[str] = []
    for file in files:
        try:
            dest = uploads_dir / (file.filename or f"upload_{uuid.uuid4().hex}")
            content_bytes = await file.read()
            dest.write_bytes(content_bytes)
            saved_paths.append(dest)
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    doc_ids = ingest_paths(
        runtime.settings,
        runtime.bot.ctx.stores,
        saved_paths,
        source_type=source_type,
        tenant_id=ctx.tenant_id,
    )

    # Copy to workspace if active
    ws_id = x_conversation_id or runtime.settings.default_conversation_id
    ws_root = runtime.settings.workspace_dir / ws_id
    workspace_copies: List[str] = []
    if ws_root.is_dir():
        for p in saved_paths:
            try:
                shutil.copy2(p, ws_root / p.name)
                workspace_copies.append(p.name)
            except Exception:
                pass

    result: Dict[str, Any] = {
        "object": "upload.result",
        "tenant_id": ctx.tenant_id,
        "ingested_count": len(doc_ids),
        "doc_ids": doc_ids,
        "filenames": [p.name for p in saved_paths],
        "errors": errors,
    }
    if workspace_copies:
        result["workspace_copies"] = workspace_copies
    return result
