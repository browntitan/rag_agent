"""Docker-based sandbox executor for safe Python code execution.

Each call to ``DockerSandboxExecutor.execute()`` creates a fresh, isolated
Docker container, runs the supplied Python code inside it, captures stdout/
stderr, and destroys the container — regardless of success or failure.

Security properties:
- Network disabled (``network_disabled=True``)
- Memory limited (default 512 MiB)
- Container auto-removed in finally block
- stdout/stderr truncated to 50 KB each
"""
from __future__ import annotations

import io
import logging
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agentic_chatbot.sandbox.exceptions import SandboxUnavailableError

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB per stream
_DEFAULT_PACKAGES = ["pandas", "openpyxl", "xlrd"]


@dataclass
class SandboxResult:
    """Output captured from a single sandbox execution."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_seconds: float
    truncated: bool = False  # True when stdout or stderr was truncated to 50 KB

    @property
    def success(self) -> bool:
        """Return True when the process exited cleanly (exit code 0)."""
        return self.exit_code == 0


class DockerSandboxExecutor:
    """Execute Python code inside a short-lived, isolated Docker container.

    Usage::

        executor = DockerSandboxExecutor(timeout_seconds=30)
        result = executor.execute(
            code="import pandas as pd; print(pd.__version__)",
            files={"/workspace/data.csv": "/host/path/data.csv"},
        )
        print(result.stdout)   # pandas version string
        print(result.success)  # True
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout_seconds: int = 60,
        memory_limit: str = "512m",
        network_disabled: bool = True,
        work_dir: str = "/workspace",
    ) -> None:
        """Store configuration. Docker connection is deferred to ``execute()``."""
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self.work_dir = work_dir

    def execute(
        self,
        code: str,
        files: Optional[Dict[str, str]] = None,
        packages: Optional[List[str]] = None,
        workspace_path: Optional[Path] = None,
    ) -> SandboxResult:
        """Run *code* in a fresh Docker container.

        Args:
            code:           Python source code to execute. Must use ``print()`` for output.
            files:          Mapping of ``{container_path: host_path}`` for files to copy in.
                            Ignored when *workspace_path* is provided (files already live there).
            packages:       Additional pip packages to install (on top of the defaults).
            workspace_path: When provided, bind-mount this host directory at ``/workspace``
                            inside the container with read-write access.  Files written
                            by the code persist on the host after the container is removed,
                            making them available to subsequent calls within the same session.
                            When ``None`` (default), the existing ``put_archive`` behaviour
                            is used (backward compatible).

        Returns:
            :class:`SandboxResult` with stdout, stderr, exit_code, timing, and
            a ``truncated`` flag set when output exceeded 50 KB.

        Raises:
            :class:`~agentic_chatbot.sandbox.exceptions.SandboxUnavailableError`:
                When Docker is not reachable on the host.
        """
        try:
            import docker  # noqa: PLC0415
        except ImportError as exc:
            raise SandboxUnavailableError(
                "The 'docker' package is not installed. Run: pip install docker>=7.0.0"
            ) from exc

        try:
            client = docker.from_env()
        except Exception as exc:  # docker.errors.DockerException and friends
            raise SandboxUnavailableError(
                f"Docker is not available: {exc}. Ensure Docker is running."
            ) from exc

        all_packages = list(_DEFAULT_PACKAGES)
        for pkg in packages or []:
            if pkg not in all_packages:
                all_packages.append(pkg)

        run_script = self._build_run_script(code, all_packages)

        # Build volumes mapping for persistent workspace bind-mount
        volumes: Optional[Dict[str, Dict[str, str]]] = None
        if workspace_path is not None:
            workspace_path.mkdir(parents=True, exist_ok=True)
            volumes = {str(workspace_path): {"bind": self.work_dir, "mode": "rw"}}
            logger.debug("Sandbox: bind-mounting workspace %s → %s", workspace_path, self.work_dir)

        container = None
        start_time = time.monotonic()
        try:
            container = client.containers.create(
                self.image,
                command=["bash", "-c", run_script],
                network_disabled=self.network_disabled,
                mem_limit=self.memory_limit,
                working_dir=self.work_dir,
                detach=True,
                auto_remove=False,
                volumes=volumes,
            )

            # Copy host files into the container before start.
            # Skipped when workspace_path is set — files already live in the bind mount.
            if files and workspace_path is None:
                self._copy_files_to_container(container, files)

            container.start()

            try:
                container.wait(timeout=self.timeout_seconds)
            except Exception as exc:
                # ReadTimeout or similar — kill the runaway container
                err_msg = str(exc)
                if "timeout" in err_msg.lower() or "ReadTimeout" in type(exc).__name__:
                    logger.warning("Sandbox container timed out after %ds", self.timeout_seconds)
                    try:
                        container.kill()
                    except Exception:
                        pass
                    elapsed = time.monotonic() - start_time
                    return SandboxResult(
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout_seconds}s.",
                        exit_code=-1,
                        execution_time_seconds=elapsed,
                    )
                raise

            elapsed = time.monotonic() - start_time
            return self._extract_output(container, elapsed)

        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _build_run_script(self, code: str, packages: List[str]) -> str:
        """Build the bash script that installs packages and runs the Python code."""
        pkg_str = " ".join(packages)
        # We write the code to a file to avoid shell escaping issues with heredoc
        # The heredoc delimiter PYTHON_EOF is unlikely to appear in user code
        script = (
            f"pip install --quiet {pkg_str} 2>/dev/null\n"
            f"mkdir -p {self.work_dir}\n"
            f"cat << 'PYTHON_EOF' > {self.work_dir}/script.py\n"
            f"{code}\n"
            f"PYTHON_EOF\n"
            f"python {self.work_dir}/script.py\n"
        )
        return script

    def _copy_files_to_container(self, container, files: Dict[str, str]) -> None:
        """Copy host files into the running container using put_archive().

        Args:
            container: Docker SDK container object (must be created, not yet started).
            files:     Mapping of ``{container_path: host_path}``.
        """
        for container_path, host_path in files.items():
            host_p = Path(host_path)
            if not host_p.exists():
                logger.warning("Sandbox file not found, skipping: %s", host_path)
                continue

            container_p = Path(container_path)
            container_dir = str(container_p.parent)
            filename = container_p.name

            # Build an in-memory tar archive
            file_data = host_p.read_bytes()
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                info = tarfile.TarInfo(name=filename)
                info.size = len(file_data)
                tf.addfile(info, io.BytesIO(file_data))
            buf.seek(0)

            try:
                container.put_archive(container_dir, buf.read())
            except Exception as exc:
                logger.warning("Failed to copy %s into container: %s", host_path, exc)

    def _extract_output(self, container, elapsed: float) -> SandboxResult:
        """Read logs from the container and return a SandboxResult."""
        truncated = False

        try:
            raw_stdout = container.logs(stdout=True, stderr=False)
            raw_stderr = container.logs(stdout=False, stderr=True)
        except Exception as exc:
            logger.warning("Could not read container logs: %s", exc)
            raw_stdout = b""
            raw_stderr = b""

        stdout_bytes = raw_stdout if isinstance(raw_stdout, bytes) else raw_stdout.encode()
        stderr_bytes = raw_stderr if isinstance(raw_stderr, bytes) else raw_stderr.encode()

        if len(stdout_bytes) > _MAX_OUTPUT_BYTES:
            stdout_bytes = stdout_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True
        if len(stderr_bytes) > _MAX_OUTPUT_BYTES:
            stderr_bytes = stderr_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        try:
            inspect = container.wait(timeout=5)
            exit_code = inspect.get("StatusCode", 0)
        except Exception:
            exit_code = 0

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_seconds=elapsed,
            truncated=truncated,
        )
