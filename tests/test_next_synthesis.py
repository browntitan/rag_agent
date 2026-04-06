from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.rag.engine import _select_evidence_docs
from agentic_chatbot_next.rag.retrieval import GradedChunk
from agentic_chatbot_next.rag.synthesis import generate_grounded_answer


def _doc(chunk_id: str, title: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": chunk_id.split("#", 1)[0],
            "title": title,
            "chunk_index": 0,
            "source_type": "kb",
        },
    )


def test_generate_grounded_answer_falls_back_to_evidence_summary_on_invalid_json():
    llm = SimpleNamespace(invoke=lambda prompt, config=None: SimpleNamespace(content="not valid json"))
    evidence_docs = [
        _doc(
            "arch#chunk0001",
            "ARCHITECTURE.md",
            "RuntimeService coordinates routing, tool execution, and session persistence for the next runtime.",
        )
    ]

    payload = generate_grounded_answer(
        llm,
        settings=SimpleNamespace(prompts_backend="local", grounded_answer_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs?",
        conversation_context="",
        evidence_docs=evidence_docs,
        callbacks=[],
    )

    assert "Key details from the retrieved documents:" in payload["answer"]
    assert "(arch#chunk0001)" in payload["answer"]
    assert payload["used_citation_ids"] == ["arch#chunk0001"]
    assert payload["warnings"] == ["LLM_JSON_PARSE_FAILED"]


def test_generate_grounded_answer_overrides_false_no_evidence_claim_when_docs_present():
    llm = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"answer":"I cannot provide this because no evidence available.","used_citation_ids":[],"followups":[],"warnings":["No evidence available in the context."],"confidence_hint":0.0}'
        )
    )
    evidence_docs = [
        _doc(
            "arch#chunk0002",
            "C4_ARCHITECTURE.md",
            "The container view shows the frontend, API layer, RuntimeKernel, and Postgres/pgvector persistence.",
        )
    ]

    payload = generate_grounded_answer(
        llm,
        settings=SimpleNamespace(prompts_backend="local", grounded_answer_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs?",
        conversation_context="",
        evidence_docs=evidence_docs,
        callbacks=[],
    )

    assert "Key details from the retrieved documents:" in payload["answer"]
    assert "(arch#chunk0002)" in payload["answer"]
    assert payload["warnings"] == ["LLM_NO_EVIDENCE_OVERRIDE"]


def test_select_evidence_docs_prefers_strong_matches_before_supplemental():
    prompt_doc = _doc("prompt#chunk0001", "TEST_QUERIES.md", "Prompt catalog")
    arch_doc_1 = _doc("arch#chunk0001", "ARCHITECTURE.md", "RuntimeService and RuntimeKernel flow")
    arch_doc_2 = _doc("arch#chunk0002", "C4_ARCHITECTURE.md", "System context and containers")

    graded = [
        GradedChunk(doc=prompt_doc, relevance=1, reason="question_echo"),
        GradedChunk(doc=arch_doc_1, relevance=3, reason="direct"),
        GradedChunk(doc=arch_doc_2, relevance=2, reason="supporting"),
    ]

    selected = _select_evidence_docs("What are the key implementation details in the architecture docs?", graded, 2)

    assert [doc.metadata["chunk_id"] for doc in selected] == [
        "arch#chunk0001",
        "arch#chunk0002",
    ]
