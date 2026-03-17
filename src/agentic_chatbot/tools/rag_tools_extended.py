"""Extended RAG specialist tools — supplements make_all_rag_tools() in rag_tools.py.

These tools add capabilities that improve RAG quality without modifying the
core 11-tool set. They are loaded via an optional try/except import in
``rag/agent.py`` so the baseline agent is unaffected if this file is absent.

Tools added
-----------
12. query_rewriter       — Rewrite ambiguous queries before the first search
13. chunk_expander       — Fetch a chunk plus its N neighbours for full context
14. document_summarizer  — Dense summary of a document without reading all chunks
15. citation_validator   — Verify that a cited chunk actually supports a claim
16. web_search_fallback  — Tavily web search when KB evidence is insufficient
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from langchain.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.stores import KnowledgeStores

logger = logging.getLogger(__name__)


def make_extended_rag_tools(
    stores: KnowledgeStores,
    session: Any,  # ChatSession — imported lazily to avoid circular imports
    *,
    judge_llm: Any,
    settings: Optional[Settings] = None,
) -> List[Any]:
    """Return extended RAG tools bound to stores, session, and judge_llm.

    These tools are appended to the list returned by :func:`make_all_rag_tools`
    inside ``run_rag_agent()``.

    Args:
        stores:    KnowledgeStores (chunk_store, doc_store, mem_store).
        session:   ChatSession — provides tenant_id and scratchpad.
        judge_llm: A LangChain chat model used for LLM-backed tools
                   (summarisation, citation validation, query rewriting).
        settings:  Optional Settings for top_k configuration.
    """
    tenant_id: str = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
    top_k_vector: int = max(1, int(getattr(settings, "rag_top_k_vector", 8)))
    web_search_enabled: bool = bool(getattr(settings, "web_search_enabled", False))
    tavily_api_key: Optional[str] = getattr(settings, "tavily_api_key", None)

    # ------------------------------------------------------------------ #
    #  12. query_rewriter                                                  #
    # ------------------------------------------------------------------ #
    @tool
    def query_rewriter(query: str, reason: str = "") -> str:
        """Rewrite an ambiguous or poorly-formed query to improve retrieval quality.

        Use this BEFORE the first search when the user's phrasing is vague,
        uses pronouns without clear referents, or mixes multiple concepts.

        Args:
            query:  The original or partially refined query.
            reason: Optional: why a rewrite is needed (for audit logging).

        Returns JSON: {"original": "...", "rewritten": "...", "changed": true/false}
        """
        try:
            from agentic_chatbot.rag.rewrite import rewrite_query

            if reason:
                logger.debug("query_rewriter called (reason=%s)", reason)

            rewritten = rewrite_query(
                judge_llm,
                settings=settings,
                question=query,
                conversation_context=reason or "",
                attempt=1,
            )
            changed = rewritten.strip() != query.strip()
            return json.dumps({"original": query, "rewritten": rewritten, "changed": changed})
        except Exception as exc:
            logger.warning("query_rewriter failed: %s", exc)
            return json.dumps({"original": query, "rewritten": query, "changed": False, "error": str(exc)})

    # ------------------------------------------------------------------ #
    #  13. chunk_expander                                                  #
    # ------------------------------------------------------------------ #
    @tool
    def chunk_expander(chunk_id: str, window: int = 2) -> str:
        """Fetch the full text of a chunk and its neighbouring chunks.

        Search results contain 500-character snippets. Use this tool when you
        need the complete surrounding context for a retrieved chunk (e.g. the
        full paragraph before/after a key clause).

        Args:
            chunk_id: The chunk_id from a previous search result.
            window:   Number of adjacent chunks to include on each side (default 2).
                      Total retrieved = up to 2*window + 1 chunks.

        Returns JSON: {"chunks": [{"chunk_id": ..., "chunk_index": ..., "content": ...}],
                       "count": N}
        """
        try:
            chunk = stores.chunk_store.get_chunk_by_id(chunk_id, tenant_id)
            if chunk is None:
                return json.dumps({"error": f"chunk_id {chunk_id!r} not found", "chunks": []})

            min_idx = max(0, chunk.chunk_index - window)
            max_idx = chunk.chunk_index + window

            neighbours = stores.chunk_store.get_chunks_by_index_range(
                chunk.doc_id, min_idx, max_idx, tenant_id
            )

            result = [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "clause_number": c.clause_number,
                    "section_title": c.section_title,
                    "content": c.content,
                }
                for c in neighbours
            ]
            return json.dumps({"chunks": result, "count": len(result)})
        except Exception as exc:
            logger.warning("chunk_expander failed: %s", exc)
            return json.dumps({"error": str(exc), "chunks": []})

    # ------------------------------------------------------------------ #
    #  14. document_summarizer                                             #
    # ------------------------------------------------------------------ #
    @tool
    def document_summarizer(doc_id: str, focus: str = "") -> str:
        """Produce a dense summary of a document without reading all chunks.

        Retrieves the top representative chunks via vector search and asks the
        judge LLM to synthesise a 300-word summary. Use this to get an overview
        before deciding which specific clauses to drill into.

        Use resolve_document first if you only have a document name.

        Args:
            doc_id: The document to summarise.
            focus:  Optional topic to focus the summary on (e.g. "payment terms").
                    Leave empty for a general overview.

        Returns JSON: {"summary": "...", "doc_id": "...", "chunk_ids_used": [...]}
        """
        try:
            search_query = focus if focus.strip() else "document overview key topics summary"
            search_top_k = min(top_k_vector * 2, 20)

            chunks = stores.chunk_store.vector_search(
                search_query,
                top_k=search_top_k,
                doc_id_filter=doc_id,
                tenant_id=tenant_id,
            )

            if not chunks:
                return json.dumps(
                    {"error": f"No chunks found for doc_id {doc_id!r}", "summary": "", "chunk_ids_used": []}
                )

            # Sort by chunk_index for coherent reading order
            chunks_sorted = sorted(
                chunks,
                key=lambda c: c.doc.metadata.get("chunk_index", 0),
            )

            chunk_ids_used = [c.doc.metadata.get("chunk_id", "") for c in chunks_sorted]

            excerpts = "\n\n---\n\n".join(
                f"[{c.doc.metadata.get('chunk_id', '')}]\n{c.doc.page_content}"
                for c in chunks_sorted
            )

            focus_clause = f"Focus: {focus}. " if focus.strip() else ""
            prompt = (
                f"Summarise the following document excerpts in approximately 300 words. "
                f"{focus_clause}"
                f"Be concise, factual, and preserve key terminology.\n\n"
                f"Document ID: {doc_id}\n\n"
                f"Excerpts:\n{excerpts}"
            )

            resp = judge_llm.invoke(prompt)
            summary = getattr(resp, "content", None) or str(resp)

            return json.dumps({
                "summary": summary.strip(),
                "doc_id": doc_id,
                "chunk_ids_used": chunk_ids_used,
            })
        except Exception as exc:
            logger.warning("document_summarizer failed: %s", exc)
            return json.dumps({"error": str(exc), "summary": "", "chunk_ids_used": []})

    # ------------------------------------------------------------------ #
    #  15. citation_validator                                              #
    # ------------------------------------------------------------------ #
    @tool
    def citation_validator(claim: str, chunk_id: str) -> str:
        """Check whether a specific chunk actually supports a given claim.

        Use before finalising your answer when you are uncertain whether a
        cited chunk genuinely backs a statement you are making.

        Args:
            claim:    The statement you want to verify
                      (e.g. "Clause 3.1 requires monthly reporting").
            chunk_id: The chunk_id of the evidence chunk.

        Returns JSON: {"supported": true/false, "confidence": 0.0-1.0,
                       "reason": "...", "chunk_id": "..."}
        """
        try:
            chunk = stores.chunk_store.get_chunk_by_id(chunk_id, tenant_id)
            if chunk is None:
                return json.dumps({
                    "supported": False,
                    "confidence": 0.0,
                    "reason": f"Chunk {chunk_id!r} not found in the knowledge base.",
                    "chunk_id": chunk_id,
                })

            prompt = (
                "You are a fact-checking assistant. Given the following text excerpt and a claim, "
                "determine whether the excerpt supports the claim.\n\n"
                f"Excerpt ({chunk_id}):\n{chunk.content}\n\n"
                f"Claim: {claim}\n\n"
                'Return ONLY valid JSON in this exact schema:\n'
                '{"supported": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}'
            )

            resp = judge_llm.invoke(prompt)
            text = getattr(resp, "content", None) or str(resp)

            from agentic_chatbot.utils.json_utils import extract_json
            obj = extract_json(text) or {}

            supported = bool(obj.get("supported", False))
            confidence = float(obj.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(obj.get("reason", "Could not parse LLM response"))

            return json.dumps({
                "supported": supported,
                "confidence": confidence,
                "reason": reason,
                "chunk_id": chunk_id,
            })
        except Exception as exc:
            logger.warning("citation_validator failed: %s", exc)
            return json.dumps({
                "supported": False,
                "confidence": 0.0,
                "reason": str(exc),
                "chunk_id": chunk_id,
            })

    # ------------------------------------------------------------------ #
    #  16. web_search_fallback (opt-in, requires TAVILY_API_KEY)           #
    # ------------------------------------------------------------------ #
    @tool
    def web_search_fallback(query: str, max_results: int = 5) -> str:
        """Search the public web when knowledge-base evidence is insufficient.

        Use ONLY as a last resort — after search_document and search_all_documents
        have returned no relevant results. Results are marked with "source": "web"
        so they are clearly distinguished from KB citations.

        Args:
            query:       The refined search query.
            max_results: Maximum number of web results (default 5, max 10).

        Returns JSON: {"results": [{"title": ..., "url": ..., "snippet": ...}],
                       "source": "web", "count": N}
        """
        if not web_search_enabled:
            return json.dumps({
                "error": "Web search is disabled. Set WEB_SEARCH_ENABLED=true to enable.",
                "results": [],
                "source": "web",
            })

        if not tavily_api_key:
            return json.dumps({
                "error": "Web search is not configured. TAVILY_API_KEY environment variable is missing.",
                "results": [],
                "source": "web",
            })

        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore[import]

            capped = max(1, min(int(max_results), 10))
            search = TavilySearchResults(max_results=capped, tavily_api_key=tavily_api_key)
            raw = search.invoke(query)

            results = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", "")[:400],
                        })

            return json.dumps({"results": results, "source": "web", "count": len(results)})
        except ImportError:
            return json.dumps({
                "error": "langchain-community is not installed. Run: pip install langchain-community[tavily]",
                "results": [],
                "source": "web",
            })
        except Exception as exc:
            logger.warning("web_search_fallback failed: %s", exc)
            return json.dumps({"error": str(exc), "results": [], "source": "web"})

    return [
        query_rewriter,
        chunk_expander,
        document_summarizer,
        citation_validator,
        web_search_fallback,
    ]
