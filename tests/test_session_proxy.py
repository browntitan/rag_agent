"""Tests for SessionProxy workspace propagation.

These tests verify that the workspace field is correctly wired through
SessionProxy so that all graph nodes (rag, utility, data_analyst) can
access the same persistent session workspace.

No LLMs, Docker, or network required.
"""
from __future__ import annotations

import pytest

from agentic_chatbot.graph.session_proxy import SessionProxy


# ── Helpers ──────────────────────────────────────────────────────────────────

class _FakeWorkspace:
    """Minimal stand-in for SessionWorkspace used to test reference equality."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.root = None


# ── SessionProxy.workspace field ─────────────────────────────────────────────

class TestSessionProxyWorkspaceField:
    """SessionProxy carries an optional workspace reference."""

    def test_workspace_defaults_to_none(self):
        proxy = SessionProxy()
        assert proxy.workspace is None

    def test_workspace_can_be_set_at_construction(self):
        ws = _FakeWorkspace("sess-abc")
        proxy = SessionProxy(workspace=ws)
        assert proxy.workspace is ws

    def test_workspace_identity_preserved(self):
        """Same object — not a copy."""
        ws = _FakeWorkspace("sess-xyz")
        proxy = SessionProxy(workspace=ws)
        assert proxy.workspace is ws  # reference equality

    def test_workspace_does_not_appear_in_repr(self):
        """repr=False keeps repr tidy for logging."""
        ws = _FakeWorkspace("sess-repr")
        proxy = SessionProxy(workspace=ws)
        assert "workspace" not in repr(proxy)

    def test_workspace_independent_of_other_fields(self):
        ws = _FakeWorkspace("sess-123")
        proxy = SessionProxy(
            session_id="sess-123",
            tenant_id="acme",
            scratchpad={"k": "v"},
            workspace=ws,
        )
        assert proxy.session_id == "sess-123"
        assert proxy.tenant_id == "acme"
        assert proxy.scratchpad == {"k": "v"}
        assert proxy.workspace is ws

    def test_none_workspace_does_not_raise(self):
        proxy = SessionProxy(workspace=None)
        assert getattr(proxy, "workspace", "missing") is None


# ── Builder propagation (unit-level, no LLM) ─────────────────────────────────

class TestBuilderPropagation:
    """Verify workspace propagates from a session-like object to SessionProxy."""

    def test_workspace_propagated_via_getattr(self):
        """Simulate builder.py: workspace=getattr(session, 'workspace', None)."""
        ws = _FakeWorkspace("sess-build")

        class _MockSession:
            session_id = "sess-build"
            tenant_id = "local-dev"
            demo_mode = False
            scratchpad = {}
            uploaded_doc_ids = []
            workspace = ws

        session = _MockSession()
        proxy = SessionProxy(
            session_id=session.session_id,
            tenant_id=session.tenant_id,
            demo_mode=bool(getattr(session, "demo_mode", False)),
            scratchpad=dict(session.scratchpad),
            uploaded_doc_ids=list(session.uploaded_doc_ids),
            workspace=getattr(session, "workspace", None),
        )
        assert proxy.workspace is ws

    def test_missing_workspace_on_session_gives_none(self):
        """Sessions without a workspace attribute fall back to None."""

        class _LegacySession:
            session_id = "legacy"
            tenant_id = "local-dev"
            demo_mode = False
            scratchpad = {}
            uploaded_doc_ids = []
            # no 'workspace' attribute

        session = _LegacySession()
        proxy = SessionProxy(
            workspace=getattr(session, "workspace", None),
        )
        assert proxy.workspace is None

    def test_none_workspace_on_session_gives_none(self):
        """Explicit None workspace propagates as None."""

        class _UnopenedSession:
            session_id = "unopened"
            workspace = None

        session = _UnopenedSession()
        proxy = SessionProxy(workspace=getattr(session, "workspace", None))
        assert proxy.workspace is None
