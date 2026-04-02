from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Citation:
    citation_id: str
    doc_id: str
    title: str
    source_type: str
    location: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Citation":
        return cls(
            citation_id=str(raw.get("citation_id") or ""),
            doc_id=str(raw.get("doc_id") or ""),
            title=str(raw.get("title") or ""),
            source_type=str(raw.get("source_type") or ""),
            location=str(raw.get("location") or ""),
            snippet=str(raw.get("snippet") or ""),
        )


@dataclass
class RetrievalSummary:
    query_used: str
    steps: int = 0
    tool_calls_used: int = 0
    tool_call_log: List[str] = field(default_factory=list)
    citations_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RetrievalSummary":
        return cls(
            query_used=str(raw.get("query_used") or ""),
            steps=int(raw.get("steps") or 0),
            tool_calls_used=int(raw.get("tool_calls_used") or 0),
            tool_call_log=[str(item) for item in (raw.get("tool_call_log") or []) if str(item)],
            citations_found=int(raw.get("citations_found") or 0),
        )


@dataclass
class RagContract:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    used_citation_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_summary: RetrievalSummary = field(default_factory=lambda: RetrievalSummary(query_used=""))
    followups: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "used_citation_ids": list(self.used_citation_ids),
            "confidence": float(self.confidence),
            "retrieval_summary": self.retrieval_summary.to_dict(),
            "followups": list(self.followups),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RagContract":
        return cls(
            answer=str(raw.get("answer") or ""),
            citations=[Citation.from_dict(dict(item)) for item in (raw.get("citations") or []) if isinstance(item, dict)],
            used_citation_ids=[str(item) for item in (raw.get("used_citation_ids") or []) if str(item)],
            confidence=float(raw.get("confidence") or 0.0),
            retrieval_summary=RetrievalSummary.from_dict(dict(raw.get("retrieval_summary") or {})),
            followups=[str(item) for item in (raw.get("followups") or []) if str(item)],
            warnings=[str(item) for item in (raw.get("warnings") or []) if str(item)],
        )
