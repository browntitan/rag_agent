from __future__ import annotations

from typing import Any, Dict, List

from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.rag.citations import build_citations
from agentic_chatbot_next.rag.retrieval import grade_chunks, retrieve_candidates
from agentic_chatbot_next.rag.synthesis import generate_grounded_answer


def run_rag_contract(
    settings: Any,
    stores: Any,
    *,
    providers: Any,
    session: Any,
    query: str,
    conversation_context: str,
    preferred_doc_ids: list[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    max_retries: int,
    callbacks: list[Any] | None = None,
    skill_context: str = "",
    task_context: str = "",
) -> RagContract:
    del max_retries, skill_context, task_context
    retrieval = retrieve_candidates(
        stores,
        query,
        tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
        preferred_doc_ids=preferred_doc_ids,
        must_include_uploads=must_include_uploads,
        top_k_vector=top_k_vector,
        top_k_keyword=top_k_keyword,
    )
    merged = list(retrieval.get("merged") or [])
    graded = grade_chunks(
        providers.judge,
        settings=settings,
        question=query,
        chunks=[chunk.doc for chunk in merged],
        callbacks=callbacks or [],
    )
    selected_docs = [grade.doc for grade in graded if grade.relevance >= 2]
    if len(selected_docs) < int(getattr(settings, "rag_min_evidence_chunks", 1)):
        selected_docs = [grade.doc for grade in graded if grade.relevance >= 1][: max(1, getattr(settings, "rag_min_evidence_chunks", 1))]

    answer_payload = generate_grounded_answer(
        providers.chat,
        settings=settings,
        question=query,
        conversation_context=conversation_context,
        evidence_docs=selected_docs,
        callbacks=callbacks or [],
    )
    citations = build_citations(selected_docs)
    used_citation_ids = [
        citation_id
        for citation_id in answer_payload.get("used_citation_ids", [])
        if citation_id in {citation.citation_id for citation in citations}
    ]
    if not used_citation_ids:
        used_citation_ids = [citation.citation_id for citation in citations[: min(4, len(citations))]]

    return RagContract(
        answer=str(answer_payload.get("answer") or ""),
        citations=citations,
        used_citation_ids=used_citation_ids,
        confidence=float(answer_payload.get("confidence_hint") or 0.0),
        retrieval_summary=RetrievalSummary(
            query_used=query,
            steps=3,
            tool_calls_used=0,
            tool_call_log=[
                f"vector:{len(retrieval.get('vector') or [])}",
                f"keyword:{len(retrieval.get('keyword') or [])}",
                f"graded:{len(graded)}",
            ],
            citations_found=len(citations),
        ),
        followups=[str(item) for item in (answer_payload.get("followups") or []) if str(item)],
        warnings=[str(item) for item in (answer_payload.get("warnings") or []) if str(item)],
    )


def coerce_rag_contract(contract: Dict[str, Any]) -> RagContract:
    return RagContract(
        answer=str(contract.get("answer") or ""),
        citations=[
            Citation.from_dict(dict(item))
            for item in (contract.get("citations") or [])
            if isinstance(item, dict)
        ],
        used_citation_ids=[str(item) for item in (contract.get("used_citation_ids") or []) if str(item)],
        confidence=float(contract.get("confidence") or 0.0),
        retrieval_summary=RetrievalSummary.from_dict(dict(contract.get("retrieval_summary") or {})),
        followups=[str(item) for item in (contract.get("followups") or []) if str(item)],
        warnings=[str(item) for item in (contract.get("warnings") or []) if str(item)],
    )


def render_rag_contract(contract: RagContract | Dict[str, Any]) -> str:
    raw = contract.to_dict() if isinstance(contract, RagContract) else dict(contract)
    answer = raw.get("answer", "")
    citations = raw.get("citations", [])
    used = set(raw.get("used_citation_ids", []))
    warnings = raw.get("warnings", [])
    followups = raw.get("followups", [])

    lines = [str(answer).strip()]
    if citations:
        lines.append("\nCitations:")
        for citation in citations:
            item = citation.to_dict() if isinstance(citation, Citation) else dict(citation)
            citation_id = item.get("citation_id", "")
            if used and citation_id not in used:
                continue
            lines.append(f"- [{citation_id}] {item.get('title', '')} ({item.get('location', '')})")
    if warnings:
        lines.append("\nWarnings: " + ", ".join(str(item) for item in warnings))
    if followups:
        lines.append("\nFollow-ups:")
        for followup in followups:
            lines.append(f"- {followup}")
    return "\n".join(lines).strip()
