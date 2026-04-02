# OpenAI-Compatible Gateway

The live FastAPI gateway is `src/agentic_chatbot/api/main.py`.

It exposes the next runtime through OpenAI-style endpoints without changing the internal
runtime contracts.

## Supported endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/ingest/documents`

## Live runtime binding

The gateway creates:

- providers via `agentic_chatbot.providers`
- a live `RuntimeService` via `agentic_chatbot_next.app.api_adapter.ApiAdapter`

The gateway does not call the legacy runtime package directly.

## Chat completions flow

`POST /v1/chat/completions`:

1. validates the requested gateway model id
2. builds a local request context using `X-Conversation-ID`
3. converts prior OpenAI-format messages into LangChain history
4. creates a `ChatSession`
5. calls `RuntimeService.process_turn(...)`
6. wraps the returned assistant text back into OpenAI-compatible JSON or SSE chunks

## Document ingest flow

`POST /v1/ingest/documents`:

1. resolves the request context
2. ingests files through `agentic_chatbot_next.rag.ingest_paths(...)`
3. opens the session workspace using the canonical session key
4. copies ingested files into that workspace
5. returns ingest metadata

This keeps upload scope aligned with later data-analyst turns.

## In-process usage

Prefer `RuntimeService`, not `ChatbotApp.create(...)`.

```python
from agentic_chatbot.config import load_settings
from agentic_chatbot.providers import build_providers
from agentic_chatbot_next.app.service import RuntimeService

settings = load_settings()
providers = build_providers(settings)
service = RuntimeService.create(settings, providers)
session = RuntimeService.create_local_session(settings, conversation_id="my-chat-001")

answer = service.process_turn(session, user_text="Summarize the auth policy.")
```

## Compatibility note

`ChatbotApp` still exists as a deprecated shim over `RuntimeService` for one compatibility
window, but it is not the recommended in-process entrypoint anymore.
