"""Session workspace — a persistent host-side directory shared across all agent turns.

Each :class:`SessionWorkspace` maps to a single directory on the host filesystem::

    data/workspaces/<session_id>/

When the data analyst agent (or any agent with workspace tools) runs code in Docker,
this directory is bind-mounted into the container at ``/workspace``.  Because the
mount persists across container restarts, files written in turn 1 are visible in
turn 2, turn 3, and so on for the duration of the session.

Lifecycle::

    # App / CLI layer:
    workspace = SessionWorkspace.for_session(session_id, settings)
    workspace.open()          # creates the directory
    try:
        # ... chat turns ...
    finally:
        workspace.close()     # removes directory; call explicitly to clean up

Files are safe to read / write from the host (Python) side and from inside the
Docker container simultaneously because Docker bind-mounts use standard kernel FS
semantics.

Security notes
--------------
- ``filename`` parameters are sanitised to prevent path traversal (``../``, absolute
  paths, null bytes).  Any violation raises :class:`WorkspacePathError`.
- The root directory is created with ``mode=0o700`` (owner-only).
- Text files are limited to 5 MiB on read to avoid accidental context flooding.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Maximum size for workspace_read to return in one call (5 MiB)
_MAX_READ_BYTES = 5 * 1024 * 1024

# Maximum filename length
_MAX_FILENAME_LEN = 128


class WorkspacePathError(ValueError):
    """Raised when a caller supplies an unsafe or invalid filename."""


def _safe_filename(filename: str) -> str:
    """Validate and sanitise a filename.

    Rules:
    - No path separators (``/``, ``\\``)
    - No null bytes
    - No leading dots (hidden files)
    - No longer than 128 chars
    - Non-empty after stripping whitespace

    Returns the stripped filename or raises :class:`WorkspacePathError`.
    """
    name = filename.strip()
    if not name:
        raise WorkspacePathError("filename must not be empty")
    if len(name) > _MAX_FILENAME_LEN:
        raise WorkspacePathError(f"filename too long (max {_MAX_FILENAME_LEN} chars)")
    if "/" in name or "\\" in name:
        raise WorkspacePathError("filename must not contain path separators")
    if "\x00" in name:
        raise WorkspacePathError("filename must not contain null bytes")
    if name.startswith(".."):
        raise WorkspacePathError("filename must not start with '..'")
    return name


@dataclass
class SessionWorkspace:
    """Persistent host-side directory shared by all agents within one session.

    Attributes:
        session_id: The session this workspace belongs to.
        root:       Absolute path on the host, e.g. ``data/workspaces/<session_id>/``.
    """

    session_id: str
    root: Path

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_session(cls, session_id: str, workspace_dir: Path) -> "SessionWorkspace":
        """Create a workspace descriptor for *session_id* under *workspace_dir*.

        Does NOT create the directory — call :meth:`open` for that.
        """
        root = workspace_dir / session_id
        return cls(session_id=session_id, root=root)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Create the workspace directory if it does not already exist."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_meta()
        logger.info("Session workspace opened: %s", self.root)

    def close(self) -> None:
        """Remove the workspace directory and all its contents.

        Safe to call even if the directory does not exist.
        """
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            logger.info("Session workspace closed and removed: %s", self.root)

    @property
    def is_open(self) -> bool:
        """Return True if the workspace directory exists on disk."""
        return self.root.exists()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def copy_file(self, src: Path, filename: Optional[str] = None) -> Path:
        """Copy *src* into the workspace.

        Args:
            src:      Source path on the host.
            filename: Target filename inside the workspace.  Defaults to ``src.name``.

        Returns:
            The destination :class:`Path` inside the workspace.
        """
        target_name = _safe_filename(filename or src.name)
        dest = self.root / target_name
        shutil.copy2(src, dest)
        logger.debug("Workspace: copied %s → %s", src, dest)
        return dest

    def write_text(self, filename: str, content: str, encoding: str = "utf-8") -> Path:
        """Write *content* to a text file in the workspace.

        Args:
            filename: File name (no path separators).
            content:  Text to write.
            encoding: File encoding (default utf-8).

        Returns:
            The :class:`Path` of the written file.
        """
        target_name = _safe_filename(filename)
        dest = self.root / target_name
        dest.write_text(content, encoding=encoding)
        logger.debug("Workspace: wrote text file %s (%d chars)", dest, len(content))
        return dest

    def read_text(self, filename: str, encoding: str = "utf-8") -> str:
        """Read a text file from the workspace.

        Args:
            filename: File name (no path separators).
            encoding: File encoding (default utf-8).

        Returns:
            File content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            WorkspacePathError: If the filename is unsafe.
        """
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if not path.exists():
            raise FileNotFoundError(f"Workspace file not found: {filename!r}")
        size = path.stat().st_size
        if size > _MAX_READ_BYTES:
            logger.warning(
                "Workspace file %s is %.1f MiB; truncating to %.1f MiB",
                filename, size / 1e6, _MAX_READ_BYTES / 1e6,
            )
            raw = path.read_bytes()[:_MAX_READ_BYTES]
            return raw.decode(encoding, errors="replace") + "\n...[truncated]"
        return path.read_text(encoding=encoding)

    def read_bytes(self, filename: str) -> bytes:
        """Read raw bytes from a workspace file."""
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if not path.exists():
            raise FileNotFoundError(f"Workspace file not found: {filename!r}")
        return path.read_bytes()

    def exists(self, filename: str) -> bool:
        """Return True if *filename* exists in the workspace."""
        try:
            target_name = _safe_filename(filename)
        except WorkspacePathError:
            return False
        return (self.root / target_name).exists()

    def list_files(self) -> List[str]:
        """Return sorted list of filenames in the workspace (excludes ``.meta``)."""
        if not self.root.exists():
            return []
        return sorted(
            f.name
            for f in self.root.iterdir()
            if f.is_file() and not f.name.startswith(".meta")
        )

    def delete_file(self, filename: str) -> bool:
        """Delete a file from the workspace.

        Returns True if the file was deleted, False if it did not exist.
        """
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if path.exists():
            path.unlink()
            logger.debug("Workspace: deleted %s", path)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_meta(self) -> None:
        """Write a ``.meta`` JSON file with session metadata."""
        meta = {
            "session_id": self.session_id,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / ".meta").write_text(json.dumps(meta), encoding="utf-8")
