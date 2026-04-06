# OpenAI-Compatible Gateway

The live FastAPI gateway is `src/agentic_chatbot_next/api/main.py`.

It exposes the next runtime through OpenAI-style endpoints without changing the internal
runtime contracts.

## Supported endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/ingest/documents`
- `POST /v1/upload`

## Live runtime binding

The gateway creates:

- providers via `agentic_chatbot_next.providers`
- a live `RuntimeService` via `agentic_chatbot_next.app.api_adapter.ApiAdapter`

The public model contract remains `GATEWAY_MODEL_ID=enterprise-agent`. `POST /v1/chat/completions`
rejects any other `model` value even if the runtime internally uses different per-agent chat or
judge model overrides.

`GET /health/ready` is KB-aware. It returns `200` only when providers are healthy and the
configured KB/docs corpus is indexed for the active collection. When coverage is missing, it
returns `503` with `reason`, `collection_id`, `missing_sources`, and `suggested_fix`.

## Chat completions flow

`POST /v1/chat/completions`:

1. validates the requested gateway model id
2. builds a local request context using `X-Conversation-ID`
3. converts prior OpenAI-format messages into LangChain history
4. creates a `ChatSession`
5. calls `RuntimeService.process_turn(...)`
6. wraps the returned assistant text back into OpenAI-compatible JSON or SSE chunks

With `stream=true`, the SSE stream currently emits named `progress` events first and then
standard OpenAI-style `chat.completion.chunk` payloads.

## Document ingest flow

`POST /v1/ingest/documents`:

1. resolves the request context
2. ingests files through `agentic_chatbot_next.rag.ingest_paths(...)`
3. opens the canonical session workspace using `SessionWorkspace.for_session(...)` with the
   `tenant:user:conversation` session id
4. copies ingested files into that workspace
5. returns ingest metadata

This keeps upload scope aligned with later data-analyst turns.

## Multipart upload flow

`POST /v1/upload`:

1. accepts multipart files from the frontend
2. saves them into `UPLOADS_DIR`
3. ingests them into the KB
4. attempts a workspace copy as a best-effort convenience for active sessions

Current implementation nuance:

- this best-effort copy checks for a legacy `WORKSPACE_DIR/<conversation_id>/` directory
  rather than opening the canonical session workspace
- `/v1/upload` is therefore useful for frontend KB ingest, but it is not equivalent to
  `/v1/ingest/documents` for deterministic workspace preparation

This endpoint is not identical to `/v1/ingest/documents`.

- use `/v1/ingest/documents` when you need guaranteed pre-chat workspace preparation from
  host-visible file paths
- use `/v1/upload` for browser-style multipart uploads during an active session
- do not rely on `/v1/upload` as the canonical workspace-seeding path

## In-process usage

Prefer `RuntimeService`.

```python
from agentic_chatbot_next.config import load_settings
from agentic_chatbot_next.providers import build_providers
from agentic_chatbot_next.app.service import RuntimeService

settings = load_settings()
providers = build_providers(settings)
service = RuntimeService.create(settings, providers)
session = RuntimeService.create_local_session(settings, conversation_id="my-chat-001")

answer = service.process_turn(session, user_text="Summarize the auth policy.")
```
