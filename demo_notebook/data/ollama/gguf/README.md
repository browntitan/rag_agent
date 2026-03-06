# GGUF Inbox for demo_notebook Ollama Container

Place GGUF files and Modelfiles in this folder.

This folder is mounted into the Ollama container at `/gguf` by
`demo_notebook/docker-compose.yml`.

Example import:

```bash
docker compose -f demo_notebook/docker-compose.yml --profile ollama up -d notebook-ollama
docker compose -f demo_notebook/docker-compose.yml exec notebook-ollama \
  ollama create qwen3-8b-local -f /gguf/Modelfile
```
