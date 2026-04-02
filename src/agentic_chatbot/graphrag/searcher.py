"""GraphRAG search — wraps ``graphrag query`` for local/global/drift search.

Local search: entity-centric (fans out from matched entities to neighbors).
Global search: holistic (uses community summaries for broad questions).
Drift search: hybrid of local + community context.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def graph_search(
    query: str,
    project_dir: Path,
    *,
    method: str = "local",
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
    timeout: int = 120,
) -> str:
    """Run a GraphRAG search query against an indexed document.

    Args:
        query: The search query.
        project_dir: Path to the GraphRAG project directory.
        method: "local" | "global" | "drift" | "basic".
        community_level: Leiden hierarchy level (higher = broader).
        response_type: Output format hint for the LLM.
        timeout: Max seconds to wait.

    Returns:
        The search result as a string, or an error message.
    """
    if not shutil.which("graphrag"):
        return "GraphRAG CLI not installed. Install with: pip install graphrag"

    output_dir = project_dir / "output"
    if not output_dir.exists():
        return f"No GraphRAG index found for this document. Index may still be building."

    cmd = [
        "graphrag", "query",
        "--root", str(project_dir),
        "--method", method,
        "--community-level", str(community_level),
        "--response-type", response_type,
        "--no-streaming",
        query,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return f"GraphRAG search failed: {result.stderr[:300]}"

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return f"GraphRAG search timed out after {timeout}s"
    except FileNotFoundError:
        return "GraphRAG CLI not found"


def list_indexed_documents(graphrag_data_dir: Path) -> List[str]:
    """List all document IDs that have completed GraphRAG indexes."""
    if not graphrag_data_dir.exists():
        return []
    return [
        d.name
        for d in graphrag_data_dir.iterdir()
        if d.is_dir() and (d / "output").exists()
    ]
