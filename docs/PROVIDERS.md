# Providers: Ollama (default) and Azure OpenAI (optional)

This repo defaults to **Ollama** because it supports local development.

## Ollama

In `.env`:

```bash
LLM_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Run:

```bash
ollama serve
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

## Azure OpenAI

If you want to use Azure for chat:

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=YOUR_CHAT_DEPLOYMENT
```

If you also want Azure embeddings:

```bash
EMBEDDINGS_PROVIDER=azure
AZURE_OPENAI_EMBED_DEPLOYMENT=YOUR_EMBED_DEPLOYMENT
```

## Where it is implemented

See `src/agentic_chatbot/providers/llm_factory.py`.

