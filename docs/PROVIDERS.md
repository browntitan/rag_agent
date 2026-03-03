# Providers and backend config

## LLM / embeddings providers

Supported today:

- chat LLM: `ollama` or `azure`
- judge LLM: `ollama` or `azure`
- embeddings: `ollama` or `azure`

### Ollama example

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

### Azure OpenAI example

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

## Storage/backend switches

```bash
DATABASE_BACKEND=postgres
VECTOR_STORE_BACKEND=pgvector
OBJECT_STORE_BACKEND=local
SKILLS_BACKEND=local
PROMPTS_BACKEND=local
```

Current implementation supports `postgres` + `pgvector` + local file-backed skills/prompts/ingestion.

`OBJECT_STORE_BACKEND=s3|azure_blob` and remote skills/prompts backends are scaffolded in config but not implemented yet.

## Path-based prompt/skills config

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
