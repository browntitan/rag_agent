# Vercel AI SDK Example

This example shows how to put a Next.js AI SDK frontend in front of the Python gateway.

## What It Assumes

- The Python gateway is already running at `http://localhost:8000/v1`
- You want the Python backend to remain the single agent model
- The frontend is only responsible for UI and transport

## Files

- `lib/agentic-provider.ts` creates the OpenAI-compatible provider
- `app/api/chat/route.ts` forwards chat traffic to the Python gateway
- `app/page.tsx` is a minimal `useChat` UI
- `.env.local.example` contains the environment variables to copy into your app

## Why `@ai-sdk/openai-compatible`

The Python backend exposes `POST /v1/chat/completions`, not `/v1/responses`, so
`@ai-sdk/openai-compatible` is the safest default.

If you use `@ai-sdk/openai`, call `createOpenAI(...).chat(...)` explicitly.

## Notes

- The route forwards a stable `X-Conversation-ID` header using the AI SDK chat id.
- File uploads are a separate backend call to `/v1/ingest/documents`.
- `forceAgent` is optional; when provided, it is forwarded as `metadata.force_agent`.
