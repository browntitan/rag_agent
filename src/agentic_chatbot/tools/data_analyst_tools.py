"""Data analyst tools — load datasets, inspect columns, execute code in Docker sandbox.

Factory function ``make_data_analyst_tools`` returns up to 11 tools available
to the data analyst agent:

1. load_dataset       — load Excel/CSV from KB, return schema + preview
2. inspect_columns    — per-column statistics for planning
3. execute_code       — run Python in a Docker sandbox
4. calculator         — quick math (reused from tools/calculator.py)
5. scratchpad_write   — within-turn memory
6. scratchpad_read    — within-turn memory
7. scratchpad_list    — within-turn memory
8. workspace_write    — write a text file to the persistent session workspace
9. workspace_read     — read a file from the persistent session workspace
10. workspace_list    — list all files in the persistent session workspace
11. search_skills     — search the skills library for operational guidance
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag import KnowledgeStores
from agentic_chatbot.sandbox.docker_executor import DockerSandboxExecutor  # noqa: E402
from agentic_chatbot.sandbox.exceptions import SandboxUnavailableError  # noqa: E402

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def make_data_analyst_tools(
    stores: KnowledgeStores,
    session: Any,
    *,
    settings: Settings,
) -> List[Any]:
    """Build and return up to 11 tools for the data analyst agent.

    Args:
        stores:   KnowledgeStores providing access to doc_store and chunk_store.
        session:  A session-like object with ``scratchpad`` and optional
                  ``workspace`` access for persistent storage.
        settings: Application settings (sandbox config, etc.).

    Returns:
        List of LangChain tool callables (7 core + 3 workspace + search_skills).
    """

    # ------------------------------------------------------------------
    # Tool 1: load_dataset
    # ------------------------------------------------------------------
    @tool
    def load_dataset(doc_id: str) -> str:
        """Load a dataset (Excel or CSV) from the knowledge base.

        Use this FIRST before any analysis. Returns the column names, shape,
        dtypes, and first 5 rows so you can understand the data before writing
        any analysis code.

        Args:
            doc_id: The document ID of the uploaded file. Use list_indexed_docs
                    or resolve_document to find the correct doc_id.

        Returns:
            JSON string with keys: file_path, doc_id, columns, shape (list of
            [nrows, ncols]), dtypes (dict col->dtype), head (first 5 rows as
            records), info_summary (human-readable shape description).
            On error, returns JSON with an "error" key.
        """
        try:
            import pandas as pd  # noqa: PLC0415

            # Look up the document record to get the source file path
            doc = stores.doc_store.get_document(doc_id, tenant_id=session.tenant_id)
            if doc is None:
                return json.dumps({"error": f"Document '{doc_id}' not found in knowledge base."})

            # Resolve the source file path
            source_path = getattr(doc, "source_path", None) or getattr(doc, "file_path", None)
            if not source_path:
                # Try to construct path from source_uri
                source_uri = getattr(doc, "source_uri", "") or ""
                if source_uri.startswith("file://"):
                    source_path = source_uri[7:]
                else:
                    return json.dumps({"error": f"Cannot resolve file path for document '{doc_id}'."})

            path = Path(str(source_path))
            if not path.exists():
                return json.dumps({"error": f"File not found on disk: {path}"})

            ext = path.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                return json.dumps({
                    "error": f"Unsupported file type '{ext}'. Supported: {', '.join(_SUPPORTED_EXTENSIONS)}"
                })

            # Read full file for shape, first 5 rows for preview
            if ext == ".csv":
                df_full = pd.read_csv(path)
                df_head = df_full.head(5)
            else:
                df_full = pd.read_excel(path)
                df_head = df_full.head(5)

            nrows, ncols = df_full.shape

            # Store path in scratchpad for other tools to reference
            session.scratchpad[f"dataset_{doc_id}"] = str(path)
            session.scratchpad[f"dataset_{doc_id}_ext"] = ext

            # Copy into the persistent session workspace so the Docker sandbox
            # can access it via the bind-mounted /workspace directory.
            workspace = getattr(session, "workspace", None)
            if workspace is not None:
                try:
                    workspace.copy_file(path)
                    logger.debug("load_dataset: copied %s into session workspace", path.name)
                except Exception as ws_exc:
                    logger.warning("load_dataset: could not copy %s to workspace: %s", path.name, ws_exc)

            # Convert head to records safely (handle non-serializable types)
            head_records = []
            for record in df_head.to_dict(orient="records"):
                safe_record = {}
                for k, v in record.items():
                    try:
                        json.dumps(v)
                        safe_record[str(k)] = v
                    except (TypeError, ValueError):
                        safe_record[str(k)] = str(v)
                head_records.append(safe_record)

            return json.dumps({
                "file_path": str(path),
                "doc_id": doc_id,
                "columns": list(df_full.columns.astype(str)),
                "shape": [nrows, ncols],
                "dtypes": {str(col): str(dtype) for col, dtype in df_full.dtypes.items()},
                "head": head_records,
                "info_summary": f"{nrows:,} rows x {ncols} columns",
            })

        except Exception as exc:
            logger.warning("load_dataset failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Tool 2: inspect_columns
    # ------------------------------------------------------------------
    @tool
    def inspect_columns(doc_id: str, columns: str = "") -> str:
        """Get detailed statistics for specific columns in a loaded dataset.

        Use this to understand data distributions, null counts, unique values,
        and basic statistics BEFORE writing analysis code.

        Args:
            doc_id:  The document ID of the dataset (must be loaded first via load_dataset).
            columns: Comma-separated column names to inspect. Leave empty for all columns.

        Returns:
            JSON string keyed by column name. Numeric columns include: count, nulls,
            unique, mean, std, min, p25, p50, p75, max. String/object columns include:
            count, nulls, unique, top_values (top 5 with frequencies).
            On error, returns JSON with an "error" key.
        """
        try:
            import pandas as pd  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            path_str = session.scratchpad.get(f"dataset_{doc_id}")
            if not path_str:
                return json.dumps({
                    "error": f"Dataset '{doc_id}' not loaded. Call load_dataset('{doc_id}') first."
                })

            path = Path(path_str)
            ext = session.scratchpad.get(f"dataset_{doc_id}_ext", path.suffix.lower())

            if ext == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            # Select columns
            if columns.strip():
                col_list = [c.strip() for c in columns.split(",") if c.strip()]
                missing = [c for c in col_list if c not in df.columns]
                if missing:
                    return json.dumps({
                        "error": f"Columns not found: {missing}. Available: {list(df.columns.astype(str))}"
                    })
                df = df[col_list]

            stats: dict = {}
            for col in df.columns:
                series = df[col]
                null_count = int(series.isna().sum())
                total_count = len(series)
                unique_count = int(series.nunique(dropna=True))

                col_key = str(col)
                if pd.api.types.is_numeric_dtype(series):
                    desc = series.describe()
                    stats[col_key] = {
                        "dtype": str(series.dtype),
                        "count": total_count,
                        "nulls": null_count,
                        "unique": unique_count,
                        "mean": _safe_float(desc.get("mean")),
                        "std": _safe_float(desc.get("std")),
                        "min": _safe_float(desc.get("min")),
                        "p25": _safe_float(desc.get("25%")),
                        "p50": _safe_float(desc.get("50%")),
                        "p75": _safe_float(desc.get("75%")),
                        "max": _safe_float(desc.get("max")),
                    }
                else:
                    top_values = (
                        series.value_counts(dropna=True)
                        .head(5)
                        .to_dict()
                    )
                    stats[col_key] = {
                        "dtype": str(series.dtype),
                        "count": total_count,
                        "nulls": null_count,
                        "unique": unique_count,
                        "top_values": {str(k): int(v) for k, v in top_values.items()},
                    }

            return json.dumps(stats)

        except Exception as exc:
            logger.warning("inspect_columns failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Tool 3: execute_code
    # ------------------------------------------------------------------
    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute Python code in a secure Docker sandbox to analyze data.

        The sandbox has pandas, openpyxl, and xlrd pre-installed. Files from
        the knowledge base are mounted at /workspace/<filename>. ALWAYS use
        print() for output — only stdout is captured and returned.

        IMPORTANT: Before using this tool, use load_dataset and inspect_columns
        to understand the data first. Write focused, correct code.

        Args:
            code:    Python code to execute. Must use print() for all output.
                     Max recommended size: ~200 lines.
            doc_ids: Comma-separated doc_ids whose dataset files should be
                     mounted in the sandbox at /workspace/<filename>.

        Returns:
            JSON string with keys: stdout, stderr, success (bool),
            execution_time_seconds, truncated (bool — True if output was
            cut at 50 KB). On Docker unavailability, returns an error key.
        """
        try:
            executor = DockerSandboxExecutor(
                image=settings.sandbox_docker_image,
                timeout_seconds=settings.sandbox_timeout_seconds,
                memory_limit=settings.sandbox_memory_limit,
            )

            workspace = getattr(session, "workspace", None)
            if workspace is not None:
                # Persistent workspace bind-mount: files already live in the directory.
                # Ensure any doc_ids that weren't copied earlier are copied now.
                for raw_id in (doc_ids or "").split(","):
                    did = raw_id.strip()
                    if not did:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{did}")
                    if path_str:
                        src = Path(path_str)
                        if not workspace.exists(src.name):
                            try:
                                workspace.copy_file(src)
                            except Exception as cp_exc:
                                logger.warning("execute_code: could not copy %s to workspace: %s", src.name, cp_exc)
                    else:
                        logger.warning("execute_code: dataset_%s not in scratchpad; did you call load_dataset?", did)

                result = executor.execute(code=code, workspace_path=workspace.root)
            else:
                # Legacy path: copy files into the container via put_archive.
                files: dict = {}
                for raw_id in (doc_ids or "").split(","):
                    did = raw_id.strip()
                    if not did:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{did}")
                    if path_str:
                        host_path = Path(path_str)
                        container_path = f"/workspace/{host_path.name}"
                        files[container_path] = str(host_path)
                    else:
                        logger.warning("execute_code: dataset_%s not in scratchpad; did you call load_dataset?", did)

                result = executor.execute(code=code, files=files or None)

            return json.dumps({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.success,
                "execution_time_seconds": round(result.execution_time_seconds, 3),
                "truncated": result.truncated,
            })

        except SandboxUnavailableError as exc:
            return json.dumps({
                "error": f"Docker sandbox is not available: {exc}",
                "success": False,
                "stdout": "",
                "stderr": "",
            })
        except Exception as exc:
            logger.warning("execute_code unexpected error: %s", exc)
            return json.dumps({
                "error": str(exc),
                "success": False,
                "stdout": "",
                "stderr": "",
            })

    # ------------------------------------------------------------------
    # Tool 4: calculator (reuse existing)
    # ------------------------------------------------------------------
    from agentic_chatbot.tools.calculator import calculator  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Tools 5-7: scratchpad (same closure pattern as rag_tools.py)
    # ------------------------------------------------------------------
    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """Save an intermediate finding, observation, or plan to the scratchpad.

        Use this to track: data overview, analysis plan, intermediate results,
        or any information you want to reference in later steps.

        Args:
            key:   A descriptive key name (e.g. "data_overview", "analysis_plan").
            value: The content to store.

        Returns:
            JSON confirmation with the key and value length.
        """
        session.scratchpad[key] = value
        return json.dumps({"saved": key, "length": len(value)})

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a previously saved value from the scratchpad.

        Args:
            key: The key to retrieve.

        Returns:
            JSON with the key and value. If not found, returns available keys.
        """
        if key in session.scratchpad:
            return json.dumps({"key": key, "value": session.scratchpad[key]})
        available = [k for k in session.scratchpad.keys() if not k.startswith("dataset_")]
        return json.dumps({"error": f"Key '{key}' not found", "available_keys": available})

    @tool
    def scratchpad_list() -> str:
        """List all user-written keys currently in the scratchpad.

        Returns:
            JSON with list of keys (excluding internal dataset_ entries) and count.
        """
        user_keys = [k for k in session.scratchpad.keys() if not k.startswith("dataset_")]
        return json.dumps({"keys": user_keys, "count": len(user_keys)})

    # ------------------------------------------------------------------
    # Tools 8-10: workspace (persistent cross-turn file storage)
    # These tools are only functional when session.workspace is set.
    # They degrade gracefully (return an informative message) when no
    # workspace is available so the same tool list works in all contexts.
    # ------------------------------------------------------------------

    @tool
    def workspace_write(filename: str, content: str) -> str:
        """Write a text file to the persistent session workspace.

        Files written here survive across turns and are visible to the
        Docker sandbox at /workspace/<filename>. Use this to save analysis
        results, summaries, or notes that you (or other agents) can read
        back later in the same session.

        Args:
            filename: Name of the file to write (e.g. "results.csv",
                      "notes.txt"). Must not contain path separators.
            content:  Text content to write.

        Returns:
            JSON with the filename and byte size on success, or an error key.
        """
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            dest = workspace.write_text(filename, content)
            return json.dumps({"written": dest.name, "size_bytes": dest.stat().st_size})
        except Exception as exc:
            logger.warning("workspace_write failed for %s: %s", filename, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def workspace_read(filename: str) -> str:
        """Read a file from the persistent session workspace.

        Use this to retrieve files written by workspace_write or produced
        by execute_code in a previous turn.

        Args:
            filename: Name of the file to read. Use workspace_list() to see
                      available files.

        Returns:
            The file contents as a string, or JSON with an error key.
        """
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            return workspace.read_text(filename)
        except FileNotFoundError:
            available = workspace.list_files()
            return json.dumps({
                "error": f"File '{filename}' not found.",
                "available_files": available,
            })
        except Exception as exc:
            logger.warning("workspace_read failed for %s: %s", filename, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def workspace_list() -> str:
        """List all files currently in the persistent session workspace.

        Returns:
            JSON with a list of filenames and the total file count.
        """
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            files = workspace.list_files()
            return json.dumps({"files": files, "count": len(files)})
        except Exception as exc:
            logger.warning("workspace_list failed: %s", exc)
            return json.dumps({"error": str(exc)})

    # skills search — lets the agent look up operational guidance at runtime
    skills_search = None
    try:
        from agentic_chatbot.tools.skills_search_tool import make_skills_search_tool  # noqa: PLC0415
        skills_search = make_skills_search_tool(settings, stores=stores, session=session)
    except Exception as e:
        logger.warning("Could not build search_skills tool: %s", e)

    tools = [
        load_dataset,
        inspect_columns,
        execute_code,
        calculator,
        scratchpad_write,
        scratchpad_read,
        scratchpad_list,
        workspace_write,
        workspace_read,
        workspace_list,
    ]
    if skills_search is not None:
        tools.append(skills_search)
    return tools


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float safely, returning None for NaN/inf."""
    try:
        import math  # noqa: PLC0415
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None
