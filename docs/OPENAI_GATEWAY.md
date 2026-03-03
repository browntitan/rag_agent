# OpenAI-Compatible Agent Gateway

This project now includes a FastAPI gateway that exposes OpenAI-compatible endpoints in front of the existing agentic runtime.

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/ingest/documents`

## Run

```bash
python run.py serve-api --host 0.0.0.0 --port 8000
```

## Auth

All `/v1/*` endpoints require:

```http
Authorization: Bearer <JWT>
```

Required JWT claims:

- `sub` (user id)
- `tenant_id`

Configured via:

- `JWT_SECRET_KEY`
- `JWT_ALGORITHM` (default `HS256`)
- `RATE_LIMIT_PER_MINUTE` (default `120`, key: `tenant_id:user_id`)

## OpenWebUI integration

1. Add a new OpenAI-compatible provider.
2. Set Base URL to `http://<gateway-host>:8000/v1`.
3. Use any API key value in UI; actual auth is your Bearer JWT at the gateway/proxy layer.
4. Select model ID `enterprise-agent` (or `GATEWAY_MODEL_ID`).

## AI SDK integration (Vercel)

Point your chat route to the gateway's OpenAI-compatible endpoint.

- Base URL: `http://<gateway-host>:8000/v1`
- Model: `enterprise-agent`
- Headers: pass Bearer JWT and optional `X-Conversation-ID`

## Notes

- Chat context in v1 is client-provided `messages` history.
- CLI/demo compatibility is preserved through local defaults:
  - `DEFAULT_TENANT_ID`
  - `DEFAULT_USER_ID`
  - `DEFAULT_CONVERSATION_ID`
