from __future__ import annotations

from pathlib import Path

from agentic_chatbot.runtime.context import SessionState, ToolContext


def test_tool_context_exposes_session_like_properties():
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
        request_id="req-a",
        uploaded_doc_ids=["doc-1"],
        demo_mode=True,
        workspace_root="/tmp/runtime-workspace",
    )
    session.scratchpad["note"] = "remember this"
    context = ToolContext(
        settings=object(),
        providers=object(),
        stores=object(),
        session=session,
    )

    assert context.session_id == "tenant-a:user-a:conv-a"
    assert context.tenant_id == "tenant-a"
    assert context.user_id == "user-a"
    assert context.conversation_id == "conv-a"
    assert context.request_id == "req-a"
    assert context.uploaded_doc_ids == ["doc-1"]
    assert context.scratchpad == {"note": "remember this"}
    assert context.demo_mode is True
    assert context.workspace_root == Path("/tmp/runtime-workspace")


def test_tool_context_rehydrates_workspace_from_session_root(tmp_path: Path):
    workspace_root = tmp_path / "workspaces" / "tenant:user:conv"
    workspace_root.mkdir(parents=True)
    context = ToolContext(
        settings=object(),
        providers=object(),
        stores=object(),
        session=SessionState(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conv",
            workspace_root=str(workspace_root),
        ),
    )

    workspace = context.workspace

    assert workspace is not None
    assert workspace.session_id == context.session_id
    assert workspace.root == workspace_root
    assert context.workspace is workspace
