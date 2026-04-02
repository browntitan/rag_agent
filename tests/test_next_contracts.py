from __future__ import annotations

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.contracts.tools import ToolDefinition


def test_contract_round_trips_preserve_fields() -> None:
    message = RuntimeMessage(
        role="assistant",
        content="hello",
        name="helper",
        tool_call_id="tool-1",
        artifact_refs=["artifact://one"],
        metadata={"agent": "general"},
    )
    restored_message = RuntimeMessage.from_dict(message.to_dict())
    assert restored_message == message

    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        request_id="request",
        session_id="session",
        messages=[message],
        uploaded_doc_ids=["doc-1"],
        scratchpad={"a": "b"},
        workspace_root="/tmp/workspace",
        active_agent="general",
        metadata={"runtime_kind": "next"},
    )
    notification = TaskNotification(job_id="job-1", status="completed", summary="done")
    session.pending_notifications.append(notification)
    restored_session = SessionState.from_dict(session.to_dict())
    assert restored_session.to_dict() == session.to_dict()

    agent = AgentDefinition(
        name="general",
        mode="react",
        description="desc",
        prompt_file="general_agent.md",
        skill_scope="general",
        allowed_tools=["calculator"],
        allowed_worker_agents=["utility"],
        preload_skill_packs=["base"],
        memory_scopes=["conversation", "user"],
        max_steps=10,
        max_tool_calls=12,
        allow_background_jobs=True,
        metadata={"role_kind": "top_level"},
    )
    assert AgentDefinition.from_dict(agent.to_dict()) == agent

    tool = ToolDefinition(
        name="calculator",
        group="utility",
        description="math",
        args_schema={"type": "object"},
        read_only=True,
        background_safe=True,
        concurrency_key="utility",
        serializer="default",
        metadata={"kind": "tool"},
    )
    assert ToolDefinition.from_dict(tool.to_dict()) == tool

    job = JobRecord(
        job_id="job-1",
        session_id="session",
        agent_name="utility",
        status="completed",
        prompt="do work",
        description="worker",
        artifact_dir="/tmp/artifacts",
        output_path="/tmp/artifacts/output.md",
        result_path="/tmp/artifacts/result.json",
        result_summary="done",
        session_state={"messages": 2},
        metadata={"background": True},
    )
    assert JobRecord.from_dict(job.to_dict()) == job

    rag = RagContract(
        answer="answer",
        citations=[
            Citation(
                citation_id="c1",
                doc_id="doc-1",
                title="Doc",
                source_type="pdf",
                location="p.1",
                snippet="snippet",
            )
        ],
        used_citation_ids=["c1"],
        confidence=0.7,
        retrieval_summary=RetrievalSummary(
            query_used="query",
            steps=2,
            tool_calls_used=3,
            tool_call_log=["search"],
            citations_found=1,
        ),
        followups=["next"],
        warnings=["warn"],
    )
    assert RagContract.from_dict(rag.to_dict()).to_dict() == rag.to_dict()


def test_runtime_message_langchain_conversion_preserves_identity_fields() -> None:
    message = RuntimeMessage(
        role="assistant",
        content="hello",
        artifact_refs=["artifact://one"],
        metadata={"agent": "general"},
    )
    restored = RuntimeMessage.from_langchain(message.to_langchain())
    assert restored.message_id == message.message_id
    assert restored.created_at == message.created_at
    assert restored.artifact_refs == message.artifact_refs
    assert restored.metadata["agent"] == "general"
