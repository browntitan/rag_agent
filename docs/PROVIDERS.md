# Providers and Backend Config

## LLM / Embeddings Providers

Supported today:

- chat LLM: `ollama` or `azure`
- judge LLM: `ollama` or `azure`
- embeddings: `ollama` or `azure`

### Ollama Example

```bash
LLM_PROVIDER=ollama
JUDGE_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gpt-oss:20b
OLLAMA_JUDGE_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.2
JUDGE_TEMPERATURE=0.0
```

### Ollama GGUF Model Workflow

GGUF is supported via Ollama model creation.

Manual flow:

1. Put your `.gguf` file and `Modelfile` under `./data/ollama/gguf`.
2. Create model in the running Ollama container:

```bash
docker compose exec ollama ollama create my-gguf-model -f /gguf/Modelfile
```

3. Point app settings to the created model:

```bash
OLLAMA_CHAT_MODEL=my-gguf-model
OLLAMA_JUDGE_MODEL=my-gguf-model
```

Optional auto-import flow:

```bash
OLLAMA_GGUF_AUTO_IMPORT=true
OLLAMA_GGUF_MODEL_NAME=my-gguf-model
OLLAMA_GGUF_MODELFILE=/gguf/Modelfile
```

When auto-import is enabled, `ollama-gguf-importer` runs once and creates the model if missing.

### Azure OpenAI Example

```bash
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=azure

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=YOUR_CHAT_DEPLOYMENT
AZURE_OPENAI_JUDGE_DEPLOYMENT=YOUR_JUDGE_DEPLOYMENT
AZURE_OPENAI_EMBED_DEPLOYMENT=YOUR_EMBED_DEPLOYMENT
AZURE_TEMPERATURE=0.2
JUDGE_TEMPERATURE=0.0
```

## Storage / Backend Switches

```bash
DATABASE_BACKEND=postgres
VECTOR_STORE_BACKEND=pgvector
OBJECT_STORE_BACKEND=local
SKILLS_BACKEND=local
PROMPTS_BACKEND=local
```

Current implementation supports `postgres` + `pgvector` + local file-backed skills/prompts/ingestion.

`OBJECT_STORE_BACKEND=s3|azure_blob` and remote skills/prompts backends are scaffolded in config but not implemented yet.

## Path-Based Prompt / Skills Config

```bash
SKILLS_DIR=./data/skills
PROMPTS_DIR=./data/prompts

SHARED_SKILLS_PATH=./data/skills/skills.md
GENERAL_AGENT_SKILLS_PATH=./data/skills/general_agent.md
RAG_AGENT_SKILLS_PATH=./data/skills/rag_agent.md
SUPERVISOR_AGENT_SKILLS_PATH=./data/skills/supervisor_agent.md
UTILITY_AGENT_SKILLS_PATH=./data/skills/utility_agent.md
BASIC_CHAT_SKILLS_PATH=./data/skills/basic_chat.md

JUDGE_GRADING_PROMPT_PATH=./data/prompts/judge_grading.txt
JUDGE_REWRITE_PROMPT_PATH=./data/prompts/judge_rewrite.txt
GROUNDED_ANSWER_PROMPT_PATH=./data/prompts/grounded_answer.txt
RAG_SYNTHESIS_PROMPT_PATH=./data/prompts/rag_synthesis.txt
PARALLEL_RAG_SYNTHESIS_PROMPT_PATH=./data/prompts/parallel_rag_synthesis.txt
```
