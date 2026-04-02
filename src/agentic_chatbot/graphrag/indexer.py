"""GraphRAG indexing — runs ``graphrag index`` as a subprocess.

The CLI is the stable public interface per GraphRAG docs. We wrap it
in a subprocess call with timeout and error handling.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def graphrag_available() -> bool:
    """Check if the ``graphrag`` CLI is installed and reachable."""
    return shutil.which("graphrag") is not None


def run_graphrag_index(
    project_dir: Path,
    *,
    method: str = "standard",
    timeout: int = 600,
) -> bool:
    """Run GraphRAG indexing on a document's project directory.

    Args:
        project_dir: Path containing settings.yaml + input/ directory.
        method: "standard" (full pipeline) or "fast" (skip some steps).
        timeout: Maximum seconds to wait (default 10 minutes).

    Returns:
        True if indexing succeeded.
    """
    if not graphrag_available():
        logger.warning("graphrag CLI not found — skipping indexing for %s", project_dir.name)
        return False

    cmd = [
        "graphrag", "index",
        "--root", str(project_dir),
        "--method", method,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error(
                "GraphRAG indexing failed for %s (exit %d): %s",
                project_dir.name, result.returncode, result.stderr[:500],
            )
            return False

        logger.info("GraphRAG indexing completed for %s", project_dir.name)
        return True

    except subprocess.TimeoutExpired:
        logger.error("GraphRAG indexing timed out (%ds) for %s", timeout, project_dir.name)
        return False
    except FileNotFoundError:
        logger.error("graphrag command not found — install with: pip install graphrag")
        return False
