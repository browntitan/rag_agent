"""Unit tests for data analyst tools (make_data_analyst_tools).

All external dependencies (stores, DockerSandboxExecutor) are mocked.
Real pandas operations are used for load_dataset and inspect_columns tests
so that the stat computation logic is validated end-to-end.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_session(scratchpad=None):
    session = MagicMock()
    session.scratchpad = scratchpad or {}
    session.tenant_id = "test-tenant"
    session.workspace = None
    return session


def _make_stores(source_path=None):
    stores = MagicMock()
    doc = MagicMock()
    doc.source_path = source_path
    doc.source_uri = f"file://{source_path}" if source_path else None
    stores.doc_store.get_document.return_value = doc
    return stores


def _make_settings():
    settings = MagicMock()
    settings.sandbox_docker_image = "python:3.12-slim"
    settings.sandbox_timeout_seconds = 30
    settings.sandbox_memory_limit = "256m"
    return settings


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content)


def _write_xlsx(path: Path) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "value", "category"])
    ws.append(["Alice", 100, "A"])
    ws.append(["Bob", 200, "B"])
    ws.append(["Carol", 150, "A"])
    wb.save(str(path))


# ---------------------------------------------------------------------------
# load_dataset — CSV
# ---------------------------------------------------------------------------

class TestLoadDatasetCsv:
    def test_returns_expected_keys(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")

        session = _make_session()
        stores = _make_stores(source_path=str(csv_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")

        result = json.loads(load_tool.invoke({"doc_id": "doc123"}))

        assert "file_path" in result
        assert "doc_id" in result
        assert "columns" in result
        assert "shape" in result
        assert "dtypes" in result
        assert "head" in result
        assert "info_summary" in result

    def test_shape_correct(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "a,b\n1,2\n3,4\n5,6\n")

        session = _make_session()
        stores = _make_stores(source_path=str(csv_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")

        result = json.loads(load_tool.invoke({"doc_id": "doc123"}))

        assert result["shape"] == [3, 2]
        assert result["columns"] == ["a", "b"]

    def test_scratchpad_populated(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "x,y\n1,2\n")

        session = _make_session()
        stores = _make_stores(source_path=str(csv_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        load_tool.invoke({"doc_id": "doc_abc"})

        assert "dataset_doc_abc" in session.scratchpad
        assert session.scratchpad["dataset_doc_abc"] == str(csv_file)

    def test_head_has_at_most_5_rows(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        rows = "\n".join([f"{i},{i*2}" for i in range(20)])
        _write_csv(csv_file, f"a,b\n{rows}")

        session = _make_session()
        stores = _make_stores(source_path=str(csv_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "doc123"}))

        assert len(result["head"]) <= 5


# ---------------------------------------------------------------------------
# load_dataset — Excel
# ---------------------------------------------------------------------------

class TestLoadDatasetXlsx:
    def test_loads_excel_file(self, tmp_path):
        pytest.importorskip("openpyxl")
        xlsx_file = tmp_path / "data.xlsx"
        _write_xlsx(xlsx_file)

        session = _make_session()
        stores = _make_stores(source_path=str(xlsx_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "xlsdoc"}))

        assert "error" not in result
        assert result["columns"] == ["name", "value", "category"]
        assert result["shape"][0] == 3  # 3 data rows


# ---------------------------------------------------------------------------
# load_dataset — error cases
# ---------------------------------------------------------------------------

class TestLoadDatasetErrors:
    def test_invalid_extension_returns_error(self, tmp_path):
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("just text")

        session = _make_session()
        stores = _make_stores(source_path=str(txt_file))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "bad_doc"}))

        assert "error" in result
        assert "Unsupported" in result["error"]

    def test_missing_document_returns_error(self):
        session = _make_session()
        stores = MagicMock()
        stores.doc_store.get_document.return_value = None
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "nonexistent"}))

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_file_not_on_disk_returns_error(self, tmp_path):
        session = _make_session()
        stores = _make_stores(source_path=str(tmp_path / "ghost.csv"))
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "ghost_doc"}))

        assert "error" in result

    def test_next_runtime_load_dataset_can_resolve_workspace_file(self, tmp_path):
        csv_file = tmp_path / "workspace_data.csv"
        _write_csv(csv_file, "region,spend\nNA,10\nEU,20\n")

        session = _make_session()
        session.workspace = MagicMock()
        session.workspace.root = tmp_path
        session.workspace.exists.side_effect = lambda filename: (tmp_path / filename).exists()
        stores = MagicMock()
        stores.doc_store.get_document.return_value = None
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools

        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({"doc_id": "workspace_data.csv"}))

        assert "error" not in result
        assert result["doc_id"] == "workspace_data.csv"
        assert result["shape"] == [2, 2]

    def test_next_runtime_load_dataset_defaults_to_first_workspace_file_when_doc_id_missing(self, tmp_path):
        csv_file = tmp_path / "workspace_data.csv"
        _write_csv(csv_file, "region,spend\nNA,10\nEU,20\n")

        session = _make_session()
        session.workspace = MagicMock()
        session.workspace.root = tmp_path
        session.workspace.exists.side_effect = lambda filename: (tmp_path / filename).exists()
        session.workspace.list_files.return_value = ["workspace_data.csv"]
        stores = MagicMock()
        stores.doc_store.get_document.return_value = None
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools

        tools = make_data_analyst_tools(stores, session, settings=settings)
        load_tool = next(t for t in tools if t.name == "load_dataset")
        result = json.loads(load_tool.invoke({}))

        assert "error" not in result
        assert result["doc_id"] == "workspace_data.csv"


# ---------------------------------------------------------------------------
# inspect_columns — numeric
# ---------------------------------------------------------------------------

class TestInspectColumnsNumeric:
    def test_numeric_columns_have_stats(self, tmp_path):
        csv_file = tmp_path / "nums.csv"
        _write_csv(csv_file, "score,value\n10,100\n20,200\n30,300\n40,400\n")

        session = _make_session()
        session.scratchpad["dataset_d1"] = str(csv_file)
        session.scratchpad["dataset_d1_ext"] = ".csv"
        stores = _make_stores()
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        inspect_tool = next(t for t in tools if t.name == "inspect_columns")
        result = json.loads(inspect_tool.invoke({"doc_id": "d1", "columns": "score"}))

        assert "score" in result
        col = result["score"]
        assert "mean" in col
        assert "std" in col
        assert "min" in col
        assert "max" in col
        assert "nulls" in col

    def test_correct_mean_computed(self, tmp_path):
        csv_file = tmp_path / "nums.csv"
        _write_csv(csv_file, "v\n10\n20\n30\n")

        session = _make_session()
        session.scratchpad["dataset_d2"] = str(csv_file)
        session.scratchpad["dataset_d2_ext"] = ".csv"
        stores = _make_stores()
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        inspect_tool = next(t for t in tools if t.name == "inspect_columns")
        result = json.loads(inspect_tool.invoke({"doc_id": "d2", "columns": "v"}))

        assert abs(result["v"]["mean"] - 20.0) < 0.001


# ---------------------------------------------------------------------------
# inspect_columns — string
# ---------------------------------------------------------------------------

class TestInspectColumnsString:
    def test_string_columns_have_top_values(self, tmp_path):
        csv_file = tmp_path / "cats.csv"
        _write_csv(csv_file, "cat\nA\nA\nB\nC\nA\n")

        session = _make_session()
        session.scratchpad["dataset_d3"] = str(csv_file)
        session.scratchpad["dataset_d3_ext"] = ".csv"
        stores = _make_stores()
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        inspect_tool = next(t for t in tools if t.name == "inspect_columns")
        result = json.loads(inspect_tool.invoke({"doc_id": "d3", "columns": "cat"}))

        assert "cat" in result
        col = result["cat"]
        assert "top_values" in col
        assert "nulls" in col
        assert "unique" in col


# ---------------------------------------------------------------------------
# inspect_columns — before load_dataset
# ---------------------------------------------------------------------------

class TestInspectColumnsBeforeLoad:
    def test_returns_error_when_not_loaded(self):
        session = _make_session()  # empty scratchpad
        stores = _make_stores()
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        inspect_tool = next(t for t in tools if t.name == "inspect_columns")
        result = json.loads(inspect_tool.invoke({"doc_id": "not_loaded", "columns": ""}))

        assert "error" in result
        assert "load_dataset" in result["error"]


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

class TestExecuteCodeSuccess:
    def test_returns_stdout_on_success(self, tmp_path):
        from agentic_chatbot_next.sandbox.docker_exec import SandboxResult

        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "a,b\n1,2\n")
        session = _make_session()
        session.scratchpad["dataset_d_exec"] = str(csv_file)
        stores = _make_stores()
        settings = _make_settings()

        mock_result = SandboxResult(
            stdout="42", stderr="", exit_code=0, execution_time_seconds=0.5
        )

        with patch(
            "agentic_chatbot_next.tools.data_analyst_tools.DockerSandboxExecutor"
        ) as MockExec:
            MockExec.return_value.execute.return_value = mock_result

            from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
            tools = make_data_analyst_tools(stores, session, settings=settings)
            exec_tool = next(t for t in tools if t.name == "execute_code")
            result = json.loads(exec_tool.invoke({
                "code": "print(42)",
                "doc_ids": "d_exec",
            }))

        assert result["stdout"] == "42"
        assert result["success"] is True


class TestExecuteCodeError:
    def test_stderr_returned_on_failure(self, tmp_path):
        from agentic_chatbot_next.sandbox.docker_exec import SandboxResult

        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "a\n1\n")
        session = _make_session()
        session.scratchpad["dataset_d_err"] = str(csv_file)
        stores = _make_stores()
        settings = _make_settings()

        mock_result = SandboxResult(
            stdout="",
            stderr="NameError: name 'x' is not defined",
            exit_code=1,
            execution_time_seconds=0.1,
        )

        with patch(
            "agentic_chatbot_next.tools.data_analyst_tools.DockerSandboxExecutor"
        ) as MockExec:
            MockExec.return_value.execute.return_value = mock_result

            from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
            tools = make_data_analyst_tools(stores, session, settings=settings)
            exec_tool = next(t for t in tools if t.name == "execute_code")
            result = json.loads(exec_tool.invoke({
                "code": "print(x)",
                "doc_ids": "d_err",
            }))

        assert result["success"] is False
        assert "NameError" in result["stderr"]


class TestExecuteCodeTimeout:
    def test_timeout_message_returned(self):
        from agentic_chatbot_next.sandbox.docker_exec import SandboxResult

        session = _make_session()
        stores = _make_stores()
        settings = _make_settings()

        timeout_result = SandboxResult(
            stdout="",
            stderr="Execution timed out after 30s.",
            exit_code=-1,
            execution_time_seconds=30.0,
        )

        with patch(
            "agentic_chatbot_next.tools.data_analyst_tools.DockerSandboxExecutor"
        ) as MockExec:
            MockExec.return_value.execute.return_value = timeout_result

            from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
            tools = make_data_analyst_tools(stores, session, settings=settings)
            exec_tool = next(t for t in tools if t.name == "execute_code")
            result = json.loads(exec_tool.invoke({"code": "import time; time.sleep(999)"}))

        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()


class TestExecuteCodeDockerUnavailable:
    def test_graceful_error_when_docker_unavailable(self):
        from agentic_chatbot_next.sandbox.exceptions import SandboxUnavailableError

        session = _make_session()
        stores = _make_stores()
        settings = _make_settings()

        with patch(
            "agentic_chatbot_next.tools.data_analyst_tools.DockerSandboxExecutor"
        ) as MockExec:
            MockExec.return_value.execute.side_effect = SandboxUnavailableError("Docker not running")

            from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
            tools = make_data_analyst_tools(stores, session, settings=settings)
            exec_tool = next(t for t in tools if t.name == "execute_code")
            result = json.loads(exec_tool.invoke({"code": "print('hello')"}))

        assert "error" in result
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Scratchpad tools
# ---------------------------------------------------------------------------

class TestScratchpadTools:
    def _get_tools(self):
        session = _make_session()
        stores = _make_stores()
        settings = _make_settings()

        from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools
        tools = make_data_analyst_tools(stores, session, settings=settings)
        write = next(t for t in tools if t.name == "scratchpad_write")
        read = next(t for t in tools if t.name == "scratchpad_read")
        lst = next(t for t in tools if t.name == "scratchpad_list")
        return write, read, lst, session

    def test_write_then_read(self):
        write, read, _, session = self._get_tools()
        write.invoke({"key": "plan", "value": "step 1: inspect"})
        result = json.loads(read.invoke({"key": "plan"}))
        assert result["value"] == "step 1: inspect"

    def test_read_missing_key_returns_available(self):
        _, read, _, session = self._get_tools()
        result = json.loads(read.invoke({"key": "nonexistent"}))
        assert "error" in result
        assert "available_keys" in result

    def test_list_returns_user_keys(self):
        write, _, lst, session = self._get_tools()
        write.invoke({"key": "data_overview", "value": "3 rows x 5 cols"})
        write.invoke({"key": "analysis_plan", "value": "group by region"})
        result = json.loads(lst.invoke({}))
        assert "data_overview" in result["keys"]
        assert "analysis_plan" in result["keys"]
        assert result["count"] >= 2
