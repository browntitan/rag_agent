# Ollama GGUF Drop Folder

Place your local GGUF model files and `Modelfile` in this folder.

Example `Modelfile`:

```text
FROM ./my-model.gguf
TEMPLATE "{{ .Prompt }}"
PARAMETER num_ctx 8192
```

Manual create flow:

```bash
docker compose exec ollama ollama create my-gguf-model -f /gguf/Modelfile
```

Optional auto-import flow (set in `.env`):

- `OLLAMA_GGUF_AUTO_IMPORT=true`
- `OLLAMA_GGUF_MODEL_NAME=my-gguf-model`
- `OLLAMA_GGUF_MODELFILE=/gguf/Modelfile`
