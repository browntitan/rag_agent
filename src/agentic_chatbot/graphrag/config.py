"""Generate GraphRAG settings.yaml from application Settings.

Each indexed document gets its own GraphRAG project directory containing
a ``settings.yaml`` and an ``input/`` folder with the document text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from agentic_chatbot.config import Settings

logger = logging.getLogger(__name__)


def generate_graphrag_settings(settings: Settings, doc_id: str) -> Path:
    """Create a GraphRAG project directory for a specific document.

    Returns the project directory path. The caller is responsible for
    writing the document text into ``<project_dir>/input/``.
    """
    project_dir = settings.graphrag_data_dir / doc_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "input").mkdir(exist_ok=True)

    # Determine whether the configured models are Ollama (ollama/...) or a
    # cloud provider.  Ollama models are served locally and need api_base set;
    # they also don't require a real API key — LiteLLM accepts the dummy
    # string "ollama" to satisfy its validation.
    _is_ollama_completion = settings.graphrag_completion_model.startswith("ollama/")
    _is_ollama_embedding = settings.graphrag_embedding_model.startswith("ollama/")

    completion_model_cfg: Dict[str, Any] = {
        "model": settings.graphrag_completion_model,
        "type": "litellm",
    }
    if _is_ollama_completion:
        completion_model_cfg["api_base"] = settings.graphrag_ollama_base_url
        completion_model_cfg["api_key"] = "ollama"
    else:
        completion_model_cfg["api_key"] = "${GRAPHRAG_API_KEY}"

    embedding_model_cfg: Dict[str, Any] = {
        "model": settings.graphrag_embedding_model,
        "type": "litellm",
    }
    if _is_ollama_embedding:
        embedding_model_cfg["api_base"] = settings.graphrag_ollama_base_url
        embedding_model_cfg["api_key"] = "ollama"
    else:
        embedding_model_cfg["api_key"] = "${GRAPHRAG_EMBEDDING_API_KEY}"

    config: Dict[str, Any] = {
        "completion_models": {
            "default": completion_model_cfg,
        },
        "embedding_models": {
            "default": embedding_model_cfg,
        },
        "input": {
            "storage": {"type": "file", "base_dir": "input"},
            "type": "text",
            "file_pattern": ".*\\.txt$",
        },
        "output": {
            "storage": {"type": "file", "base_dir": "output"},
        },
        "chunking": {
            "type": "tokens",
            "size": settings.graphrag_chunk_size,
            "overlap": settings.graphrag_chunk_overlap,
        },
        "extract_graph": {
            "entity_types": [
                "PERSON", "ORGANIZATION", "LOCATION", "DATE",
                "MONETARY_AMOUNT", "LEGAL_TERM", "CLAUSE", "DOCUMENT",
            ],
            "max_gleanings": 1,
        },
        "cluster_graph": {
            "max_cluster_size": 10,
        },
        "vector_store": {
            "type": "lancedb",
            "db_uri": str(project_dir / "lancedb"),
        },
    }

    yaml_path = project_dir / "settings.yaml"

    try:
        import yaml as pyyaml  # noqa: PLC0415
        yaml_path.write_text(pyyaml.dump(config, default_flow_style=False))
    except ImportError:
        # Fallback: write as JSON (GraphRAG also accepts JSON settings)
        import json  # noqa: PLC0415
        yaml_path = project_dir / "settings.json"
        yaml_path.write_text(json.dumps(config, indent=2))

    logger.info("GraphRAG settings written to %s", yaml_path)
    return project_dir
