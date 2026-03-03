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

If you use Docker Compose, the `app` service now starts this gateway automatically.

## Authentication

The simplified gateway mode has no built-in auth on `/v1/*`.

For production, protect it upstream using one of:

- reverse proxy auth (for example Nginx/Traefik/OAuth2 proxy),
- API gateway auth/policies,
- private network-only exposure.

## OpenWebUI integration

1. Add a new OpenAI-compatible provider.
2. Set Base URL to `http://<gateway-host>:8000/v1`.
3. Use any API key value in UI if required by the client.
4. Select model ID `enterprise-agent` (or `GATEWAY_MODEL_ID`).

## AI SDK integration (Vercel)

Point your chat route to the gateway's OpenAI-compatible endpoint.

- Base URL: `http://<gateway-host>:8000/v1`
- Model: `enterprise-agent`
- Headers: optional `X-Conversation-ID`

## Notes

- Chat context in v1 is client-provided `messages` history.
- CLI/demo compatibility is preserved through local defaults:
  - `DEFAULT_TENANT_ID`
  - `DEFAULT_USER_ID`
  - `DEFAULT_CONVERSATION_ID`
