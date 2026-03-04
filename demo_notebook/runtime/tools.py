from __future__ import annotations

import json
import math
import re
from typing import Callable, Dict, List

from langchain.tools import tool

from .config import NotebookSettings
from .stores import PostgresVectorStore


def _safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _parse_csv(values: str) -> List[str]:
    if not values.strip():
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def make_calculator_tool() -> Callable:
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression for basic arithmetic."""
        safe_expr = expression.strip()
        if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.%]+", safe_expr):
            return "Invalid expression: only numbers and + - * / ( ) % are allowed."
        try:
            # Restricted eval for demo-only arithmetic.
            result = eval(safe_expr, {"__builtins__": {}}, {})
        except Exception as exc:
            return f"Calculation error: {exc}"
        if isinstance(result, float) and math.isfinite(result):
            return f"{result:.6g}"
        return str(result)

    return calculator


def make_list_docs_tool(store: PostgresVectorStore) -> Callable:
    @tool
    def list_indexed_docs() -> str:
        """List all indexed knowledge-base documents."""
        docs = store.list_documents()
        grouped = {
            "contracts": [],
            "security_compliance": [],
            "runbooks": [],
            "api_references": [],
            "other": [],
        }
        for d in docs:
            t = d.title.lower()
            if "runbook" in t or "playbook" in t:
                grouped["runbooks"].append(d.title)
            elif t.startswith("api_") or "api" in t:
                grouped["api_references"].append(d.title)
            elif any(k in t for k in ["agreement", "contract", "addendum", "schedule"]):
                grouped["contracts"].append(d.title)
            elif any(k in t for k in ["security", "privacy", "compliance", "control", "incident"]):
                grouped["security_compliance"].append(d.title)
            else:
                grouped["other"].append(d.title)
        return _safe_json({"count": len(docs), "groups": grouped})

    return list_indexed_docs


def make_rag_tools(store: PostgresVectorStore, settings: NotebookSettings) -> List[Callable]:
    @tool
    def resolve_document(title_hint: str) -> str:
        """Resolve a document title hint to likely document IDs."""
        hits = store.search_titles(title_hint, limit=6)
        return _safe_json(hits)

    @tool
    def search_document(doc_id: str, query: str, top_k: int = 6) -> str:
        """Search one document using hybrid retrieval and return scored chunks."""
        rows = store.hybrid_search(
            query,
            top_k_vector=min(top_k, settings.rag_top_k_vector),
            top_k_keyword=min(top_k, settings.rag_top_k_keyword),
            doc_id=doc_id,
        )
        payload = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "title": r.get("title", ""),
                "score": round(float(r.get("hybrid_score", 0.0)), 4),
                "clause_number": r.get("clause_number"),
                "snippet": r["content"][:500],
            }
            for r in rows[:top_k]
        ]
        return _safe_json(payload)

    @tool
    def search_all_documents(query: str, top_k: int = 10) -> str:
        """Search across all indexed documents using hybrid retrieval."""
        rows = store.hybrid_search(
            query,
            top_k_vector=min(top_k, settings.rag_top_k_vector),
            top_k_keyword=min(top_k, settings.rag_top_k_keyword),
        )
        payload = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "title": r.get("title", ""),
                "score": round(float(r.get("hybrid_score", 0.0)), 4),
                "clause_number": r.get("clause_number"),
                "snippet": r["content"][:400],
            }
            for r in rows[:top_k]
        ]
        return _safe_json(payload)

    @tool
    def list_document_structure(doc_id: str, max_items: int = 30) -> str:
        """List structured chunk markers (clause_number/section_title) for a document."""
        rows = store.get_document_chunks(doc_id)
        out = []
        for r in rows:
            if r.get("clause_number") or r.get("section_title"):
                out.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "clause_number": r.get("clause_number"),
                        "section_title": r.get("section_title"),
                    }
                )
            if len(out) >= max_items:
                break
        return _safe_json(out)

    @tool
    def extract_clauses(doc_id: str, clause_numbers_csv: str) -> str:
        """Extract specific clause-number chunks from a document."""
        targets = set(_parse_csv(clause_numbers_csv))
        rows = store.get_document_chunks(doc_id)
        hits = [
            {
                "chunk_id": r["chunk_id"],
                "clause_number": r.get("clause_number"),
                "section_title": r.get("section_title"),
                "snippet": r["content"][:700],
            }
            for r in rows
            if r.get("clause_number") in targets
        ]
        return _safe_json(hits)

    @tool
    def diff_documents(doc_id_a: str, doc_id_b: str) -> str:
        """Diff clause identifiers between two documents."""
        a = store.get_document_chunks(doc_id_a)
        b = store.get_document_chunks(doc_id_b)
        a_clauses = sorted({x.get("clause_number") for x in a if x.get("clause_number")})
        b_clauses = sorted({x.get("clause_number") for x in b if x.get("clause_number")})

        only_a = sorted(set(a_clauses) - set(b_clauses))
        only_b = sorted(set(b_clauses) - set(a_clauses))
        shared = sorted(set(a_clauses) & set(b_clauses))

        return _safe_json(
            {
                "doc_id_a": doc_id_a,
                "doc_id_b": doc_id_b,
                "only_a": only_a,
                "only_b": only_b,
                "shared": shared,
            }
        )

    @tool
    def compare_clauses(doc_id_a: str, doc_id_b: str, clause_number: str) -> str:
        """Compare one clause between two documents and return snippets."""
        a_rows = [x for x in store.get_document_chunks(doc_id_a) if x.get("clause_number") == clause_number]
        b_rows = [x for x in store.get_document_chunks(doc_id_b) if x.get("clause_number") == clause_number]
        payload = {
            "clause_number": clause_number,
            "doc_a": [{"chunk_id": r["chunk_id"], "snippet": r["content"][:700]} for r in a_rows],
            "doc_b": [{"chunk_id": r["chunk_id"], "snippet": r["content"][:700]} for r in b_rows],
        }
        return _safe_json(payload)

    return [
        resolve_document,
        search_document,
        search_all_documents,
        list_document_structure,
        extract_clauses,
        diff_documents,
        compare_clauses,
    ]


def make_rag_agent_tool(rag_answer_fn: Callable[[str], str]) -> Callable:
    @tool
    def rag_agent_tool(query: str) -> str:
        """Run the specialist RAG agent and return a grounded answer."""
        return rag_answer_fn(query)

    return rag_agent_tool
