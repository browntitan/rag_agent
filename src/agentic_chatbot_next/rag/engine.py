from __future__ import annotations

import re
from typing import Any, Dict, List

from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.rag.citations import build_citations
from agentic_chatbot_next.rag.ingest import KBCoverageStatus, get_kb_coverage_status
from agentic_chatbot_next.rag.retrieval import GradedChunk, grade_chunks, retrieve_candidates
from agentic_chatbot_next.rag.synthesis import generate_grounded_answer


def _title_overlap_score(question: str, doc: Any) -> int:
    title = str((getattr(doc, "metadata", {}) or {}).get("title") or "").lower()
    if not title:
        return 0
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.replace("_", " ")))
    overlap = len(q_terms & t_terms)
    if "architecture" in q_terms and "architecture" in t_terms:
        overlap += 2
    return overlap


def _select_evidence_docs(question: str, graded: list[GradedChunk], min_chunks: int) -> list[Any]:
    target = max(1, int(min_chunks))
    strong = [item.doc for item in graded if item.relevance >= 2]
    strong.sort(key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    if len(strong) >= target:
        return strong

    supplemental = [item.doc for item in graded if item.relevance == 1]
    supplemental.sort(key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    return (strong + supplemental)[:target]


def _kb_not_ready_answer(status: KBCoverageStatus) -> Dict[str, Any]:
    if status.sync_error:
        detail = (
            f"Startup KB sync failed for collection '{status.collection_id}': {status.sync_error}. "
            f"Run `{status.suggested_fix}` and retry the request."
        )
        warning = "KB_SYNC_FAILED"
    else:
        detail = (
            f"The configured knowledge base is not indexed for collection '{status.collection_id}'. "
            f"Run `{status.suggested_fix}` and retry the request."
        )
        warning = "KB_COVERAGE_MISSING"

    if status.missing_source_paths:
        preview = ", ".join(status.missing_source_paths[:3])
        if len(status.missing_source_paths) > 3:
            preview += ", ..."
        detail = f"{detail} Missing sources: {preview}"

    return {
        "answer": detail,
        "used_citation_ids": [],
        "followups": [],
        "warnings": [warning],
        "confidence_hint": 0.0,
    }


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
    collection_id = str(getattr(settings, "default_collection_id", "default") or "default")
    retrieval = retrieve_candidates(
        stores,
        query,
        tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
        preferred_doc_ids=preferred_doc_ids,
        must_include_uploads=must_include_uploads,
        top_k_vector=top_k_vector,
        top_k_keyword=top_k_keyword,
        collection_id_filter=collection_id,
    )
    merged = list(retrieval.get("merged") or [])
    graded = grade_chunks(
        providers.judge,
        settings=settings,
        question=query,
        chunks=[chunk.doc for chunk in merged],
        callbacks=callbacks or [],
    )
    selected_docs = _select_evidence_docs(
        query,
        graded,
        int(getattr(settings, "rag_min_evidence_chunks", 1)),
    )

    if not selected_docs:
        kb_status = get_kb_coverage_status(
            settings,
            stores,
            tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
            collection_id=collection_id,
        )
        if not kb_status.ready:
            answer_payload = _kb_not_ready_answer(kb_status)
        else:
            answer_payload = generate_grounded_answer(
                providers.chat,
                settings=settings,
                question=query,
                conversation_context=conversation_context,
                evidence_docs=selected_docs,
                callbacks=callbacks or [],
            )
    else:
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
