"""Loop-based RAG agent — powered by LangGraph.

The RAG agent is a proper tool-calling loop agent equipped with 11 specialist
RAG tools.  It autonomously decides which documents to search, accumulates
evidence in a scratchpad, and synthesises a final citation-backed answer.

The former hand-written ``while`` loop has been replaced with
``langgraph.prebuilt.create_react_agent``, which provides the same
agent→tools→agent cycle through a compiled ``StateGraph``.  All callers
(``rag_agent_tool``, orchestrator) are unaffected because the function
signature and return type are unchanged.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_chatbot.config import Settings
from agentic_chatbot.prompting import load_rag_synthesis_prompt, render_template
from agentic_chatbot.rag.answer import build_citations, generate_grounded_answer
from agentic_chatbot.rag.stores import KnowledgeStores

logger = logging.getLogger(__name__)


def run_rag_agent(
    settings: Settings,
    stores: KnowledgeStores,
    *,
    llm: Any,
    judge_llm: Any,
    query: str,
    conversation_context: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    max_retries: int,
    session: Any,       # ChatSession — avoids circular import; accessed as session.scratchpad
    callbacks: Any = None,
) -> Dict[str, Any]:
    """Loop-based RAG agent using LangGraph ReAct agent.

    Steps:
      1. Load system prompt from skills/rag_agent.md (with fallback).
      2. Build the 11 RAG specialist tools bound to stores + session.
      3. Run the LangGraph ReAct agent (tool-calling loop).
         - The agent autonomously calls search/extract/compare/scratchpad tools.
         - Continues until the LLM produces no more tool calls or the
           recursion limit is reached.
      4. Final synthesis call: ask the LLM to produce the standard RAG
         contract JSON.
      5. Return the backward-compatible contract dict.
    """
    from agentic_chatbot.tools.rag_tools import make_all_rag_tools

    callbacks = callbacks or []

    # ------------------------------------------------------------------
    # 1. System prompt (from skills file or default)
    # ------------------------------------------------------------------
    try:
        from agentic_chatbot.rag.skills import load_rag_agent_skills
        system_prompt = load_rag_agent_skills(settings)
    except Exception:
        system_prompt = _DEFAULT_RAG_SYSTEM

    # ------------------------------------------------------------------
    # 2. Build tools (core 11 + optional extended tools)
    # ------------------------------------------------------------------
    rag_tools = make_all_rag_tools(stores, session, settings=settings)

    try:
        from agentic_chatbot.tools.rag_tools_extended import make_extended_rag_tools
        extended = make_extended_rag_tools(
            stores, session, judge_llm=judge_llm, settings=settings
        )
        rag_tools = rag_tools + extended
    except ImportError:
        pass

    try:
        llm_with_tools = llm.bind_tools(rag_tools)
    except Exception:
        llm_with_tools = None

    # ------------------------------------------------------------------
    # 3. Initial messages
    # ------------------------------------------------------------------
    task_msg = (
        f"QUERY: {query}\n"
        f"CONVERSATION_CONTEXT: {conversation_context or '(none)'}\n"
        f"PREFERRED_DOC_IDS: {preferred_doc_ids or '(search all)'}\n"
        f"MUST_INCLUDE_UPLOADS: {must_include_uploads}\n"
        f"TOP_K_VECTOR: {top_k_vector}  TOP_K_KEYWORD: {top_k_keyword}\n\n"
        "Use the available tools to retrieve evidence, then produce your final answer."
    )
    msgs: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_msg),
    ]

    # ------------------------------------------------------------------
    # 4. Tool-calling loop via LangGraph ReAct agent
    # ------------------------------------------------------------------
    steps = 0
    tool_calls_used = 0
    max_steps = settings.max_rag_agent_steps
    max_tool_calls = settings.max_tool_calls
    warnings: List[str] = []
    tool_call_log: List[str] = []

    if llm_with_tools is not None:
        from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

        graph = create_react_agent(llm, tools=rag_tools)

        # Each ReAct cycle = 2 graph-node visits (agent → tools).
        # Allow enough headroom for max_steps LLM calls and max_tool_calls.
        recursion_limit = (max_steps + max_tool_calls + 1) * 2 + 1

        try:
            result = graph.invoke(
                {"messages": msgs},
                config={"callbacks": callbacks, "recursion_limit": recursion_limit},
            )
            msgs = result["messages"]
        except Exception as e:
            # GraphRecursionError (budget exceeded) or LLM invocation error.
            logger.warning("LangGraph RAG agent stopped: %s", e)
            warnings.append(f"LANGGRAPH_AGENT_STOPPED: {str(e)[:120]}")
            # msgs retains its last known state; proceed to synthesis below.

        # Count steps and tool calls from the returned message history.
        tool_calls_used = sum(1 for m in msgs if isinstance(m, ToolMessage))
        steps = sum(1 for m in msgs if isinstance(m, AIMessage))

        # Reconstruct tool_call_log from AIMessages that carry tool_calls.
        for m in msgs:
            if isinstance(m, AIMessage):
                for tc in (getattr(m, "tool_calls", None) or []):
                    name = (
                        tc.get("name", "") if isinstance(tc, dict)
                        else getattr(tc, "name", "")
                    )
                    args = (
                        tc.get("args", {}) if isinstance(tc, dict)
                        else getattr(tc, "args", {})
                    )
                    tool_call_log.append(f"{name}({json.dumps(args)[:80]})")

    else:
        # Fallback: single-pass retrieval when tool-calling not supported
        warnings.append("TOOL_CALLING_UNSUPPORTED_FALLBACK")
        from agentic_chatbot.rag.retrieval import retrieve_candidates
        from agentic_chatbot.rag.grading import grade_chunks

        retrieval = retrieve_candidates(
            stores, query,
            tenant_id=getattr(session, "tenant_id", settings.default_tenant_id),
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=must_include_uploads,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
        )
        docs = [sc.doc for sc in retrieval.get("merged", [])]
        graded = grade_chunks(
            judge_llm,
            settings=settings,
            question=query,
            chunks=docs,
            max_chunks=12,
            callbacks=callbacks,
        )
        fallback_docs = [g.doc for g in graded if g.relevance >= 2]

        citations = build_citations(fallback_docs)
        answer_bundle = generate_grounded_answer(
            llm, question=query,
            conversation_context=conversation_context,
            evidence_docs=fallback_docs,
            max_evidence=8,
            settings=settings,
            callbacks=callbacks,
        )
        return _build_contract(answer_bundle, citations, query, warnings)

    # ------------------------------------------------------------------
    # 5. Final synthesis — ask the LLM to produce the RAG contract JSON
    # ------------------------------------------------------------------
    synthesis_prompt = render_template(
        load_rag_synthesis_prompt(settings),
        {"ORIGINAL_QUERY": query},
    )
    msgs.append(HumanMessage(content=synthesis_prompt))

    try:
        synth_resp = llm.invoke(msgs, config={"callbacks": callbacks})
        synth_text = getattr(synth_resp, "content", None) or str(synth_resp)
    except Exception as e:
        synth_text = ""
        warnings.append(f"SYNTHESIS_ERROR: {str(e)[:120]}")

    # Parse synthesis JSON
    from agentic_chatbot.utils.json_utils import coerce_float, extract_json
    answer_bundle: Dict[str, Any] = {}
    try:
        obj = extract_json(synth_text)
        if obj and isinstance(obj.get("answer"), str):
            answer_bundle = {
                "answer":            obj.get("answer", "").strip(),
                "used_citation_ids": [str(x) for x in (obj.get("used_citation_ids") or []) if str(x)],
                "followups":         [str(x) for x in (obj.get("followups") or []) if str(x)],
                "warnings":          [str(x) for x in (obj.get("warnings") or []) if str(x)],
                "confidence_hint":   coerce_float(obj.get("confidence_hint"), default=0.5),
            }
    except Exception:
        pass

    if not answer_bundle.get("answer"):
        # Use raw synthesis text as fallback answer
        answer_bundle["answer"] = synth_text.strip() or (
            "I searched the documents but could not find a confident answer. "
            "Please refine your query or check that the relevant documents are uploaded."
        )
        answer_bundle.setdefault("confidence_hint", 0.2)
        warnings.append("SYNTHESIS_JSON_PARSE_FAILED")

    # Collect all chunks retrieved by the tool loop for citation building.
    retrieved_docs = _extract_docs_from_messages(msgs)
    citations = build_citations(retrieved_docs)

    return _build_contract(
        answer_bundle, citations, query, warnings,
        tool_call_log=tool_call_log,
        steps=steps,
        tool_calls_used=tool_calls_used,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_RAG_SYSTEM = (
    "You are a specialist RAG (Retrieval-Augmented Generation) agent.\n\n"
    "Your job is to answer the user's QUERY using ONLY evidence retrieved from the indexed documents.\n\n"
    "Operating rules:\n"
    "1. ALWAYS start by calling resolve_document if the user refers to a document by name.\n"
    "2. Use list_document_structure to understand a document's outline before extracting clauses.\n"
    "3. Use search_document for targeted single-doc search; search_all_documents for broad search.\n"
    "4. Use extract_clauses when the user asks about specific numbered clauses.\n"
    "5. Use extract_requirements when asked to 'find all requirements' or similar.\n"
    "6. Use compare_clauses and diff_documents for cross-document comparison tasks.\n"
    "7. Use scratchpad_write to store intermediate results before a final synthesis.\n"
    "8. If evidence is insufficient after multiple searches, clearly state what is missing.\n"
    "9. NEVER fabricate document content — only report what the tools return.\n"
    "10. Cite inline using (chunk_id) from the tool results.\n"
)


def _build_contract(
    answer_bundle: Dict[str, Any],
    citations: List[Any],
    query: str,
    warnings: List[str],
    *,
    tool_call_log: List[str] = (),
    steps: int = 0,
    tool_calls_used: int = 0,
) -> Dict[str, Any]:
    """Assemble the backward-compatible RAG tool contract dict."""
    all_warnings = list(warnings) + answer_bundle.get("warnings", [])
    used_ids = set(answer_bundle.get("used_citation_ids", []))
    if not used_ids:
        ans_text = answer_bundle.get("answer", "")
        for c in citations:
            if c.citation_id and c.citation_id in ans_text:
                used_ids.add(c.citation_id)

    conf = float(answer_bundle.get("confidence_hint", 0.5))
    if len(citations) >= 4:
        conf = min(0.95, conf + 0.1)
    if not citations:
        conf = min(conf, 0.25)

    return {
        "answer":             answer_bundle.get("answer", ""),
        "citations":          [
            {
                "citation_id": c.citation_id,
                "doc_id":      c.doc_id,
                "title":       c.title,
                "source_type": c.source_type,
                "location":    c.location,
                "snippet":     c.snippet,
            }
            for c in citations
        ],
        "used_citation_ids":  sorted(used_ids),
        "confidence":         conf,
        "retrieval_summary":  {
            "query_used":       query,
            "steps":            steps,
            "tool_calls_used":  tool_calls_used,
            "tool_call_log":    list(tool_call_log),
            "citations_found":  len(citations),
        },
        "followups":          answer_bundle.get("followups", []),
        "warnings":           all_warnings,
    }


def _extract_docs_from_messages(msgs: List[Any]) -> List[Any]:
    """Extract LangChain Document objects from ToolMessage results in the conversation."""
    from langchain_core.documents import Document
    import json as _json

    docs: List[Document] = []
    for m in msgs:
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = _json.loads(m.content)
        except Exception:
            continue
        # ToolMessages from search tools are lists of chunk dicts
        items = data if isinstance(data, list) else data.get("requirements", data.get("chunks", []))
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            snippet = item.get("snippet", "")
            if not snippet:
                continue
            docs.append(
                Document(
                    page_content=snippet,
                    metadata={
                        "chunk_id":      item.get("chunk_id", ""),
                        "doc_id":        item.get("doc_id", ""),
                        "chunk_type":    item.get("chunk_type", "general"),
                        "clause_number": item.get("clause_number"),
                        "section_title": item.get("section_title"),
                        "page":          item.get("page_number"),
                        "title":         item.get("title", ""),
                        "source_type":   item.get("source_type", ""),
                    },
                )
            )
    return docs
