from __future__ import annotations

from typing import Any, Dict


def render_rag_contract(contract: Dict[str, Any]) -> str:
    """Render a RAG contract dict into a user-facing string."""
    ans = contract.get("answer", "")
    citations = contract.get("citations", [])
    used = set(contract.get("used_citation_ids", []))
    warnings = contract.get("warnings", [])
    followups = contract.get("followups", [])

    lines = [ans.strip()]

    if citations:
        lines.append("\nCitations:")
        for citation in citations:
            citation_id = citation.get("citation_id", "")
            if used and citation_id not in used:
                continue
            title = citation.get("title", "")
            location = citation.get("location", "")
            lines.append(f"- [{citation_id}] {title} ({location})")

    if warnings:
        lines.append("\nWarnings: " + ", ".join(str(item) for item in warnings))

    if followups:
        lines.append("\nFollow-ups:")
        for followup in followups:
            lines.append(f"- {followup}")

    return "\n".join(lines).strip()
