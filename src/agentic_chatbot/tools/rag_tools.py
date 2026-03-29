"""RAG specialist tools for the loop-based RAG agent.

All tools are created via make_all_rag_tools() which binds them to the
current KnowledgeStores and ChatSession instances.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

from langchain.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.stores import KnowledgeStores

logger = logging.getLogger(__name__)


def _chunk_to_dict(ch: Any) -> dict:
    """Convert a ChunkRecord or ScoredChunk to a JSON-serialisable dict."""
    from agentic_chatbot.db.chunk_store import ChunkRecord, ScoredChunk
    if isinstance(ch, ScoredChunk):
        meta = ch.doc.metadata or {}
        return {
            "chunk_id":      meta.get("chunk_id", ""),
            "doc_id":        meta.get("doc_id", ""),
            "chunk_type":    meta.get("chunk_type", "general"),
            "clause_number": meta.get("clause_number"),
            "section_title": meta.get("section_title"),
            "page_number":   meta.get("page"),
            "score":         round(ch.score, 4),
            "method":        ch.method,
            "snippet":       ch.doc.page_content[:500],
        }
    if isinstance(ch, ChunkRecord):
        return {
            "chunk_id":      ch.chunk_id,
            "doc_id":        ch.doc_id,
            "chunk_type":    ch.chunk_type,
            "clause_number": ch.clause_number,
            "section_title": ch.section_title,
            "page_number":   ch.page_number,
            "snippet":       ch.content[:500],
        }
    return {}


def make_all_rag_tools(
    stores: KnowledgeStores,
    session: Any,   # ChatSession — imported lazily to avoid circular imports
    *,
    settings: Settings | None = None,
) -> List[Any]:
    """Return all 14 RAG specialist tools bound to stores and session.

    Args:
        stores:   KnowledgeStores with chunk_store, doc_store, memory_store.
        session:  ChatSession — provides session.scratchpad dict.
        settings: Settings — required for the search_skills tool.
    """
    top_k_vector = max(1, int(getattr(settings, "rag_top_k_vector", 8)))
    top_k_keyword = max(1, int(getattr(settings, "rag_top_k_keyword", 8)))
    tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))

    # ------------------------------------------------------------------ #
    #  1. resolve_document                                                 #
    # ------------------------------------------------------------------ #
    @tool
    def resolve_document(name_or_hint: str) -> str:
        """Fuzzy-match a user's document name/description to indexed doc_ids.

        Steps:
          1. Exact case-insensitive title match.
          2. Fuzzy substring match via rapidfuzz WRatio scorer.
          3. pg_trgm similarity search via DocumentStore.fuzzy_search_title().

        Returns JSON: {"candidates": [{"doc_id": "...", "title": "...", "score": 0.95}]}
        """
        try:
            from rapidfuzz import fuzz
            from rapidfuzz import process as fuzz_proc
        except ImportError:
            # Fallback to pg_trgm only if rapidfuzz is not installed
            results = stores.doc_store.fuzzy_search_title(name_or_hint, tenant_id=tenant_id, limit=5)
            return json.dumps({"candidates": results})

        all_docs = stores.doc_store.get_all_titles(tenant_id=tenant_id)
        if not all_docs:
            return json.dumps({"candidates": []})

        titles = [d["title"] for d in all_docs]
        doc_map = {d["title"]: d for d in all_docs}

        # rapidfuzz returns (match_string, score, index) tuples; scores are 0-100
        fuzzy_hits = fuzz_proc.extract(
            name_or_hint, titles, scorer=fuzz.WRatio, limit=5
        )

        candidates = []
        seen: set = set()
        for title, score, _ in fuzzy_hits:
            if title in seen:
                continue
            seen.add(title)
            d = doc_map[title]
            candidates.append({
                "doc_id":      d["doc_id"],
                "title":       title,
                "source_type": d.get("source_type", ""),
                "score":       round(score / 100.0, 3),
            })

        # Also add any pg_trgm hits not already present
        trgm_hits = stores.doc_store.fuzzy_search_title(name_or_hint, tenant_id=tenant_id, limit=5)
        for h in trgm_hits:
            if h["doc_id"] not in {c["doc_id"] for c in candidates}:
                candidates.append({
                    "doc_id":      h["doc_id"],
                    "title":       h["title"],
                    "source_type": h.get("source_type", ""),
                    "score":       round(float(h.get("score", 0)), 3),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps({"candidates": candidates[:5]})

    # ------------------------------------------------------------------ #
    #  2. search_document                                                  #
    # ------------------------------------------------------------------ #
    @tool
    def search_document(doc_id: str, query: str, strategy: str = "hybrid") -> str:
        """Search within a single specific document.

        Args:
          doc_id:   The exact doc_id to search in (use resolve_document first if unsure).
          query:    The search query.
          strategy: 'vector' | 'keyword' | 'hybrid' (default: hybrid).

        Returns JSON list of matching chunks with snippet, score, clause_number.
        """
        results = []
        if strategy in ("vector", "hybrid"):
            v_hits = stores.chunk_store.vector_search(
                query,
                top_k=top_k_vector,
                doc_id_filter=doc_id,
                tenant_id=tenant_id,
            )
            results.extend(v_hits)
        if strategy in ("keyword", "hybrid"):
            k_hits = stores.chunk_store.keyword_search(
                query,
                top_k=top_k_keyword,
                doc_id_filter=doc_id,
                tenant_id=tenant_id,
            )
            results.extend(k_hits)

        # Deduplicate
        seen_ids: set = set()
        unique = []
        for r in results:
            cid = (r.doc.metadata or {}).get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(r)

        unique.sort(key=lambda x: x.score, reverse=True)
        max_out = min(50, top_k_vector + top_k_keyword)
        return json.dumps([_chunk_to_dict(r) for r in unique[:max_out]])

    # ------------------------------------------------------------------ #
    #  3. search_all_documents                                             #
    # ------------------------------------------------------------------ #
    @tool
    def search_all_documents(query: str, strategy: str = "hybrid") -> str:
        """Search across ALL indexed documents (no doc_id filter).

        Args:
          query:    The search query.
          strategy: 'vector' | 'keyword' | 'hybrid' (default: hybrid).

        Returns JSON list of top chunks with doc_id, title, snippet, score.
        """
        results = []
        if strategy in ("vector", "hybrid"):
            results.extend(stores.chunk_store.vector_search(query, top_k=top_k_vector, tenant_id=tenant_id))
        if strategy in ("keyword", "hybrid"):
            results.extend(stores.chunk_store.keyword_search(query, top_k=top_k_keyword, tenant_id=tenant_id))

        seen_ids: set = set()
        unique = []
        for r in results:
            cid = (r.doc.metadata or {}).get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(r)

        unique.sort(key=lambda x: x.score, reverse=True)
        max_out = min(80, top_k_vector + top_k_keyword)
        return json.dumps([_chunk_to_dict(r) for r in unique[:max_out]])

    # ------------------------------------------------------------------ #
    #  4. full_text_search_document                                        #
    # ------------------------------------------------------------------ #
    @tool
    def full_text_search_document(doc_id: str, query: str, max_results: int = 20) -> str:
        """Deep full-text keyword search within ONE document, returning FULL chunk content.

        Unlike search_document (which returns 500-char snippets), this returns the
        complete text of each matching chunk along with page numbers, section titles,
        and clause numbers. Use this when you need to read the full text of passages
        matching a keyword, not just previews.

        Args:
            doc_id:      Exact doc_id (use resolve_document first if needed).
            query:       Keywords to search for within the document.
            max_results: Maximum number of chunks to return (default 20, max 50).

        Returns JSON list of chunks with full content, page_number, clause_number, section_title.
        """
        capped = min(max(1, max_results), 50)
        chunks = stores.chunk_store.full_text_search_document(
            query, doc_id, top_k=capped, tenant_id=tenant_id,
        )
        results = []
        for ch in chunks:
            results.append({
                "chunk_id":      ch.chunk_id,
                "doc_id":        ch.doc_id,
                "chunk_index":   ch.chunk_index,
                "chunk_type":    ch.chunk_type,
                "clause_number": ch.clause_number,
                "section_title": ch.section_title,
                "page_number":   ch.page_number,
                "content":       ch.content,
            })
        return json.dumps(results)

    # ------------------------------------------------------------------ #
    #  5. search_by_metadata                                               #
    # ------------------------------------------------------------------ #
    @tool
    def search_by_metadata(
        source_type: str = "",
        file_type: str = "",
        title_contains: str = "",
    ) -> str:
        """Search for documents by metadata fields (source_type, file_type, title substring).

        Use this to discover what documents are available before searching their content.
        Filters are combined with AND logic — only documents matching ALL non-empty
        filters are returned.

        Args:
            source_type:    Filter by source type, e.g. 'kb' or 'upload'. Empty = any.
            file_type:      Filter by file type, e.g. 'pdf', 'docx'. Empty = any.
            title_contains: Case-insensitive substring match on document title. Empty = any.

        Returns JSON list of matching documents with doc_id, title, source_type, file_type.
        """
        all_docs = stores.doc_store.list_documents(
            source_type=source_type, tenant_id=tenant_id,
        )
        # Apply additional filters in Python
        filtered = all_docs
        if file_type:
            ft_lower = file_type.lower()
            filtered = [d for d in filtered if getattr(d, "file_type", "").lower() == ft_lower]
        if title_contains:
            tc_lower = title_contains.lower()
            filtered = [d for d in filtered if tc_lower in getattr(d, "title", "").lower()]

        results = []
        for d in filtered:
            results.append({
                "doc_id":      d.doc_id,
                "title":       d.title,
                "source_type": d.source_type,
                "file_type":   getattr(d, "file_type", ""),
                "num_chunks":  getattr(d, "num_chunks", 0),
                "ingested_at": getattr(d, "ingested_at", ""),
            })
        return json.dumps(results)

    # ------------------------------------------------------------------ #
    #  6. extract_clauses                                                  #
    # ------------------------------------------------------------------ #
    @tool
    def extract_clauses(doc_id: str, clause_numbers: str) -> str:
        """Retrieve specific numbered clauses from a document by their clause_number metadata.

        Args:
          doc_id:         The document to extract from.
          clause_numbers: Comma-separated clause numbers, e.g. '3,3.1,10.2'.

        Returns JSON list of clause dicts with full content and metadata.
        """
        nums = [n.strip() for n in clause_numbers.split(",") if n.strip()]
        if not nums:
            return json.dumps({"error": "No clause numbers provided."})
        chunks = stores.chunk_store.get_chunks_by_clause(doc_id, nums, tenant_id=tenant_id)
        if not chunks:
            return json.dumps({
                "message": f"No clauses found for numbers {nums} in doc {doc_id}. "
                           "Try list_document_structure to see available clause numbers.",
                "chunks": [],
            })
        return json.dumps([_chunk_to_dict(ch) for ch in chunks])

    # ------------------------------------------------------------------ #
    #  5. list_document_structure                                          #
    # ------------------------------------------------------------------ #
    @tool
    def list_document_structure(doc_id: str) -> str:
        """Return the clause/section outline of a document.

        Useful to understand what clauses exist before extracting specific ones,
        or to plan a clause-by-clause comparison.

        Returns JSON list of {clause_number, section_title, chunk_type} ordered by position.
        """
        outline = stores.chunk_store.get_structure_outline(doc_id, tenant_id=tenant_id)
        if not outline:
            return json.dumps({
                "message": f"No structured outline found for doc {doc_id}. "
                           "This document may be 'general' type (no detected clause structure).",
                "outline": [],
            })
        return json.dumps({"doc_id": doc_id, "outline": outline})

    # ------------------------------------------------------------------ #
    #  6. extract_requirements                                             #
    # ------------------------------------------------------------------ #
    @tool
    def extract_requirements(doc_id: str, requirement_filter: str = "") -> str:
        """Find all requirements in a document.

        Requirements are chunks tagged chunk_type='requirement' at ingest time
        (content matched shall/must/REQ-NNN patterns).

        Args:
          doc_id:             The document to search.
          requirement_filter: Optional free-text filter — semantically re-ranks results
                              to surface requirements most relevant to this description.
                              Leave empty to return all requirements.

        Returns JSON list of requirement chunks ordered by relevance or position.
        """
        semantic_q = requirement_filter.strip() or None
        chunks = stores.chunk_store.get_requirement_chunks(
            doc_id,
            semantic_query=semantic_q,
            top_k=50,
            tenant_id=tenant_id,
        )
        if not chunks:
            return json.dumps({
                "message": f"No requirement chunks found in doc {doc_id}. "
                           "The document may not contain shall/must/REQ-NNN language, "
                           "or may not have been ingested with the current pipeline.",
                "requirements": [],
            })
        return json.dumps({
            "doc_id": doc_id,
            "total": len(chunks),
            "requirements": [_chunk_to_dict(ch) for ch in chunks],
        })

    # ------------------------------------------------------------------ #
    #  7. compare_clauses                                                  #
    # ------------------------------------------------------------------ #
    @tool
    def compare_clauses(doc_id_1: str, doc_id_2: str, clause_numbers: str) -> str:
        """Side-by-side comparison of specific clauses between two documents.

        Args:
          doc_id_1:       First document.
          doc_id_2:       Second document.
          clause_numbers: Comma-separated clause numbers to compare, e.g. '3,5,10'.

        Returns JSON with doc_1_clauses, doc_2_clauses, missing_in_1, missing_in_2.
        """
        nums = [n.strip() for n in clause_numbers.split(",") if n.strip()]
        chunks_1 = stores.chunk_store.get_chunks_by_clause(doc_id_1, nums, tenant_id=tenant_id)
        chunks_2 = stores.chunk_store.get_chunks_by_clause(doc_id_2, nums, tenant_id=tenant_id)

        nums_1 = {ch.clause_number for ch in chunks_1 if ch.clause_number}
        nums_2 = {ch.clause_number for ch in chunks_2 if ch.clause_number}

        return json.dumps({
            "doc_1_clauses":  [_chunk_to_dict(ch) for ch in chunks_1],
            "doc_2_clauses":  [_chunk_to_dict(ch) for ch in chunks_2],
            "missing_in_1":   sorted(nums_2 - nums_1),
            "missing_in_2":   sorted(nums_1 - nums_2),
            "shared":         sorted(nums_1 & nums_2),
        })

    # ------------------------------------------------------------------ #
    #  8. diff_documents                                                   #
    # ------------------------------------------------------------------ #
    @tool
    def diff_documents(doc_id_1: str, doc_id_2: str) -> str:
        """Structural diff of two documents — compare their clause/section outlines.

        Identifies which clauses/sections exist in one document but not the other,
        and which are shared (by clause_number).

        Returns JSON: {shared, only_in_doc_1, only_in_doc_2, doc_1_outline, doc_2_outline}.
        """
        outline_1 = stores.chunk_store.get_structure_outline(doc_id_1, tenant_id=tenant_id)
        outline_2 = stores.chunk_store.get_structure_outline(doc_id_2, tenant_id=tenant_id)

        nums_1 = {row["clause_number"] for row in outline_1 if row.get("clause_number")}
        nums_2 = {row["clause_number"] for row in outline_2 if row.get("clause_number")}

        return json.dumps({
            "doc_1_id":      doc_id_1,
            "doc_2_id":      doc_id_2,
            "shared":        sorted(nums_1 & nums_2),
            "only_in_doc_1": sorted(nums_1 - nums_2),
            "only_in_doc_2": sorted(nums_2 - nums_1),
            "doc_1_outline": outline_1,
            "doc_2_outline": outline_2,
        })

    # ------------------------------------------------------------------ #
    #  9-11. Scratchpad tools                                              #
    # ------------------------------------------------------------------ #
    # Resolve workspace artifacts directory for persistent scratchpad
    _artifacts_dir = None
    if hasattr(session, "workspace") and session.workspace is not None:
        _ws_root = getattr(session.workspace, "root", None)
        if _ws_root is not None:
            _artifacts_dir = Path(_ws_root) / ".artifacts"

    @tool
    def scratchpad_write(key: str, value: str, persist: bool = False) -> str:
        """Write a key-value pair to the agent scratchpad.

        Use this to store intermediate findings between tool calls so you can
        build up a complete answer across multiple searches.

        Args:
            key: A descriptive key (e.g., "doc1_findings", "comparison_matrix").
            value: The content to store.
            persist: If True, also write to session workspace for cross-turn
                     persistence. Use this for findings you'll need in follow-up turns.
        """
        session.scratchpad[key] = value
        suffix = ""
        if persist and _artifacts_dir is not None:
            try:
                _artifacts_dir.mkdir(parents=True, exist_ok=True)
                (_artifacts_dir / f"{key}.md").write_text(value, encoding="utf-8")
                suffix = " [persisted to workspace]"
            except Exception as e:
                suffix = f" [persist failed: {e}]"
        return f"Stored key '{key}' in scratchpad ({len(value)} chars).{suffix}"

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a value from the scratchpad by key.

        Checks in-memory scratchpad first, then falls back to persisted
        workspace artifacts from previous turns.
        """
        val = session.scratchpad.get(key)
        if val is not None:
            return val

        # Fall back to persisted artifacts
        if _artifacts_dir is not None:
            artifact_path = _artifacts_dir / f"{key}.md"
            if artifact_path.exists():
                val = artifact_path.read_text(encoding="utf-8")
                session.scratchpad[key] = val  # cache in memory
                return val

        return f"Key '{key}' not found in scratchpad. Available keys: {list(session.scratchpad.keys())}"

    @tool
    def scratchpad_list() -> str:
        """List all keys in the scratchpad (in-memory + persisted artifacts)."""
        keys = set(session.scratchpad.keys())
        if _artifacts_dir is not None and _artifacts_dir.exists():
            for f in _artifacts_dir.glob("*.md"):
                keys.add(f.stem)
        keys_list = sorted(keys)
        return json.dumps({"keys": keys_list, "count": len(keys_list)})

    # ------------------------------------------------------------------ #
    #  GraphRAG tools (conditional — only when GRAPHRAG_ENABLED=true)     #
    # ------------------------------------------------------------------ #
    graph_search_local_tool = None
    graph_search_global_tool = None

    if settings is not None and getattr(settings, "graphrag_enabled", False):
        @tool
        def graph_search_local(query: str, doc_id: str = "") -> str:
            """Search the knowledge graph for entity-specific information.

            Use LOCAL search when the user asks about specific entities:
            "What does Company X do?", "Who is John Smith?", "What are the
            obligations under Clause 5?", "What entities are mentioned?"

            Local search fans out from matched entities to their neighbors,
            returning entity descriptions, relationships, and supporting text.

            Args:
                query: The question to answer using entity-centric graph traversal.
                doc_id: Optional — limit search to a specific document's graph.
                        If empty, searches across all indexed documents.
            """
            from agentic_chatbot.graphrag.searcher import graph_search, list_indexed_documents  # noqa: PLC0415

            if doc_id:
                project_dir = settings.graphrag_data_dir / doc_id
                if not (project_dir / "output").exists():
                    return f"No GraphRAG index found for doc_id={doc_id}. The index may still be building."
                return graph_search(query, project_dir, method="local")

            # Search across all indexed documents
            indexed = list_indexed_documents(settings.graphrag_data_dir)
            if not indexed:
                return "No documents have been indexed with GraphRAG yet."

            results = []
            for did in indexed[:3]:  # limit to 3 docs to avoid timeout
                project_dir = settings.graphrag_data_dir / did
                result = graph_search(query, project_dir, method="local")
                if result and "failed" not in result.lower():
                    results.append(f"[{did}]: {result}")

            return "\n\n---\n\n".join(results) if results else "No results found in knowledge graph."

        @tool
        def graph_search_global(query: str, doc_id: str = "") -> str:
            """Search the knowledge graph for holistic/thematic information.

            Use GLOBAL search when the user asks broad questions:
            "What are the main themes?", "Summarize all key relationships",
            "What topics does this document cover?", "Give an overview"

            Global search uses community summaries (hierarchical clustering)
            to answer questions requiring understanding of the whole document.

            Args:
                query: The broad question to answer using community summaries.
                doc_id: Optional — limit to a specific document's graph.
            """
            from agentic_chatbot.graphrag.searcher import graph_search, list_indexed_documents  # noqa: PLC0415

            if doc_id:
                project_dir = settings.graphrag_data_dir / doc_id
                if not (project_dir / "output").exists():
                    return f"No GraphRAG index found for doc_id={doc_id}. The index may still be building."
                return graph_search(query, project_dir, method="global")

            indexed = list_indexed_documents(settings.graphrag_data_dir)
            if not indexed:
                return "No documents have been indexed with GraphRAG yet."

            results = []
            for did in indexed[:3]:
                project_dir = settings.graphrag_data_dir / did
                result = graph_search(query, project_dir, method="global")
                if result and "failed" not in result.lower():
                    results.append(f"[{did}]: {result}")

            return "\n\n---\n\n".join(results) if results else "No results found in knowledge graph."

        graph_search_local_tool = graph_search_local
        graph_search_global_tool = graph_search_global

    # skills search — lets the agent look up operational guidance at runtime
    skills_search = None
    if settings is not None:
        try:
            from agentic_chatbot.tools.skills_search_tool import make_skills_search_tool  # noqa: PLC0415
            skills_search = make_skills_search_tool(settings)
        except Exception as e:
            logger.warning("Could not build search_skills tool: %s", e)

    tools = [
        resolve_document,
        search_document,
        search_all_documents,
        full_text_search_document,
        search_by_metadata,
        extract_clauses,
        list_document_structure,
        extract_requirements,
        compare_clauses,
        diff_documents,
        scratchpad_write,
        scratchpad_read,
        scratchpad_list,
    ]
    if graph_search_local_tool is not None:
        tools.append(graph_search_local_tool)
    if graph_search_global_tool is not None:
        tools.append(graph_search_global_tool)
    if skills_search is not None:
        tools.append(skills_search)
    return tools
