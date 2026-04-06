from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.tools import tool

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.sandbox.exceptions import SandboxUnavailableError
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
from agentic_chatbot_next.tools.calculator import calculator
from agentic_chatbot_next.tools.skills_search_tool import make_skills_search_tool

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def make_data_analyst_tools(
    stores: KnowledgeStores,
    session: Any,
    *,
    settings: Settings,
) -> List[Any]:
    def _first_workspace_dataset_name() -> str:
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return ""
        for filename in workspace.list_files():
            if Path(filename).suffix.lower() in _SUPPORTED_EXTENSIONS:
                return filename
        return ""

    def _first_loaded_dataset_ref() -> str:
        for key in sorted(session.scratchpad.keys()):
            if key.startswith("dataset_") and not key.endswith("_ext"):
                return key[len("dataset_") :]
        return ""

    def _resolve_dataset_path(dataset_ref: str) -> tuple[Path | None, str]:
        doc = stores.doc_store.get_document(dataset_ref, tenant_id=session.tenant_id)
        if doc is not None:
            source_path = getattr(doc, "source_path", None) or getattr(doc, "file_path", None)
            if not source_path:
                source_uri = getattr(doc, "source_uri", "") or ""
                if source_uri.startswith("file://"):
                    source_path = source_uri[7:]
            if source_path:
                return Path(str(source_path)), dataset_ref

        workspace = getattr(session, "workspace", None)
        if workspace is not None:
            candidate_name = Path(str(dataset_ref)).name
            if workspace.exists(candidate_name):
                return workspace.root / candidate_name, candidate_name

        return None, dataset_ref

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset (Excel or CSV) from the knowledge base."""
        try:
            import pandas as pd

            doc_id = str(doc_id or "").strip() or _first_workspace_dataset_name()
            if not doc_id:
                return json.dumps({"error": "No dataset reference was provided and no uploaded workspace dataset is available."})
            path, resolved_ref = _resolve_dataset_path(doc_id)
            if path is None:
                return json.dumps({"error": f"Dataset '{doc_id}' not found in the knowledge base or session workspace."})
            if not path.exists():
                return json.dumps({"error": f"File not found on disk: {path}"})
            ext = path.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                return json.dumps({"error": f"Unsupported file type '{ext}'."})

            df_full = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
            df_head = df_full.head(5)
            session.scratchpad[f"dataset_{resolved_ref}"] = str(path)
            session.scratchpad[f"dataset_{resolved_ref}_ext"] = ext

            workspace = getattr(session, "workspace", None)
            if workspace is not None and path.parent != workspace.root:
                try:
                    workspace.copy_file(path)
                except Exception as exc:
                    logger.warning("Could not copy %s into workspace: %s", path.name, exc)

            head_records = []
            for record in df_head.to_dict(orient="records"):
                safe_record = {}
                for key, value in record.items():
                    try:
                        json.dumps(value)
                        safe_record[str(key)] = value
                    except (TypeError, ValueError):
                        safe_record[str(key)] = str(value)
                head_records.append(safe_record)

            nrows, ncols = df_full.shape
            return json.dumps(
                {
                    "file_path": str(path),
                    "doc_id": resolved_ref,
                    "columns": list(df_full.columns.astype(str)),
                    "shape": [nrows, ncols],
                    "dtypes": {str(col): str(dtype) for col, dtype in df_full.dtypes.items()},
                    "head": head_records,
                    "info_summary": f"{nrows:,} rows x {ncols} columns",
                }
            )
        except Exception as exc:
            logger.warning("load_dataset failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Get detailed statistics for specific columns in a loaded dataset."""
        try:
            import pandas as pd

            doc_id = str(doc_id or "").strip() or _first_loaded_dataset_ref() or _first_workspace_dataset_name()
            if not doc_id:
                return json.dumps({"error": "No dataset reference was provided and no loaded dataset is available."})
            path_str = session.scratchpad.get(f"dataset_{doc_id}")
            if not path_str:
                return json.dumps({"error": f"Dataset '{doc_id}' not loaded. Call load_dataset first."})
            path = Path(path_str)
            ext = session.scratchpad.get(f"dataset_{doc_id}_ext", path.suffix.lower())
            df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)

            if columns.strip():
                col_list = [column.strip() for column in columns.split(",") if column.strip()]
                missing = [column for column in col_list if column not in df.columns]
                if missing:
                    return json.dumps({"error": f"Columns not found: {missing}. Available: {list(df.columns.astype(str))}"})
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
                    top_values = series.value_counts(dropna=True).head(5).to_dict()
                    stats[col_key] = {
                        "dtype": str(series.dtype),
                        "count": total_count,
                        "nulls": null_count,
                        "unique": unique_count,
                        "top_values": {str(key): int(value) for key, value in top_values.items()},
                    }
            return json.dumps(stats)
        except Exception as exc:
            logger.warning("inspect_columns failed for doc_id=%s: %s", doc_id, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute Python code in a secure Docker sandbox to analyze data."""
        try:
            executor = DockerSandboxExecutor(
                image=settings.sandbox_docker_image,
                timeout_seconds=settings.sandbox_timeout_seconds,
                memory_limit=settings.sandbox_memory_limit,
            )
            workspace = getattr(session, "workspace", None)
            if workspace is not None:
                for raw_id in (doc_ids or "").split(","):
                    doc_id = raw_id.strip()
                    if not doc_id:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{doc_id}")
                    if path_str:
                        source = Path(path_str)
                        if not workspace.exists(source.name):
                            try:
                                workspace.copy_file(source)
                            except Exception as exc:
                                logger.warning("Could not copy %s to workspace: %s", source.name, exc)
                result = executor.execute(code=code, workspace_path=workspace.root)
            else:
                files: dict = {}
                for raw_id in (doc_ids or "").split(","):
                    doc_id = raw_id.strip()
                    if not doc_id:
                        continue
                    path_str = session.scratchpad.get(f"dataset_{doc_id}")
                    if path_str:
                        host_path = Path(path_str)
                        files[f"/workspace/{host_path.name}"] = str(host_path)
                result = executor.execute(code=code, files=files or None)

            return json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.success,
                    "execution_time_seconds": round(result.execution_time_seconds, 3),
                    "truncated": result.truncated,
                }
            )
        except SandboxUnavailableError as exc:
            return json.dumps({"error": f"Docker sandbox is not available: {exc}", "success": False, "stdout": "", "stderr": ""})
        except Exception as exc:
            logger.warning("execute_code unexpected error: %s", exc)
            return json.dumps({"error": str(exc), "success": False, "stdout": "", "stderr": ""})

    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """Save an intermediate finding or plan to the scratchpad."""
        session.scratchpad[key] = value
        return json.dumps({"saved": key, "length": len(value)})

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a previously saved value from the scratchpad."""
        if key in session.scratchpad:
            return json.dumps({"key": key, "value": session.scratchpad[key]})
        available = [item for item in session.scratchpad.keys() if not item.startswith("dataset_")]
        return json.dumps({"error": f"Key '{key}' not found", "available_keys": available})

    @tool
    def scratchpad_list() -> str:
        """List all user-written keys currently in the scratchpad."""
        user_keys = [item for item in session.scratchpad.keys() if not item.startswith("dataset_")]
        return json.dumps({"keys": user_keys, "count": len(user_keys)})

    @tool
    def workspace_write(filename: str, content: str) -> str:
        """Write a text file to the persistent session workspace."""
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
        """Read a file from the persistent session workspace."""
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            return workspace.read_text(filename)
        except FileNotFoundError:
            return json.dumps({"error": f"File '{filename}' not found.", "available_files": workspace.list_files()})
        except Exception as exc:
            logger.warning("workspace_read failed for %s: %s", filename, exc)
            return json.dumps({"error": str(exc)})

    @tool
    def workspace_list() -> str:
        """List all files currently in the persistent session workspace."""
        workspace = getattr(session, "workspace", None)
        if workspace is None:
            return json.dumps({"error": "No session workspace is available."})
        try:
            files = workspace.list_files()
            return json.dumps({"files": files, "count": len(files)})
        except Exception as exc:
            logger.warning("workspace_list failed: %s", exc)
            return json.dumps({"error": str(exc)})

    skills_search = None
    try:
        skills_search = make_skills_search_tool(settings, stores=stores, session=session)
    except Exception as exc:
        logger.warning("Could not build search_skills tool: %s", exc)

    tools: List[Any] = [
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
    try:
        import math

        coerced = float(value)
        if math.isnan(coerced) or math.isinf(coerced):
            return None
        return round(coerced, 6)
    except (TypeError, ValueError):
        return None
