"""RAG synthesizer node — merges results from parallel RAG workers.

If only one worker returned, passes the result through.  For multiple
workers, uses an LLM call to produce a coherent merged answer with
combined citations.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, SystemMessage

from agentic_chatbot.graph.nodes.rag_node import render_rag_contract
from agentic_chatbot.graph.state import AgentState

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """You are merging results from multiple parallel document searches.

Below are the individual results from each worker.  Each worker searched a different
document or scope.  Your job is to:

1. Combine the findings into a single coherent answer
2. Preserve ALL citations from every worker (use inline (chunk_id) references)
3. Highlight differences and similarities between documents
4. Note any gaps or asymmetries (one doc has content the other doesn't)
5. Include warnings from any worker

{worker_results}

Produce a clear, structured response that addresses the user's original question.
Use headings or bullet points to organise cross-document comparisons."""


def make_rag_synthesizer_node(
    chat_llm: Any,
    callbacks: Optional[List[Any]] = None,
) -> Callable[[AgentState], Dict[str, Any]]:
    """Create the RAG synthesizer node function."""

    def rag_synthesizer_node(state: AgentState) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = state.get("rag_results", [])

        if not results:
            logger.warning("RAG synthesizer received no results")
            return {
                "messages": [AIMessage(content="No results were returned from document search.")],
            }

        # Single result — pass through without synthesis LLM call
        if len(results) == 1:
            contract = results[0].get("contract", {})
            rendered = render_rag_contract(contract)
            return {
                "messages": [AIMessage(content=rendered)],
            }

        # Multiple results — LLM synthesis
        worker_text_parts = []
        all_citations = []
        all_warnings = []

        for r in results:
            worker_id = r.get("worker_id", "unknown")
            contract = r.get("contract", {})
            answer = contract.get("answer", "(no answer)")
            citations = contract.get("citations", [])
            warnings = contract.get("warnings", [])

            all_citations.extend(citations)
            all_warnings.extend(warnings)

            worker_text_parts.append(
                f"### Worker: {worker_id}\n"
                f"**Query:** {r.get('query', 'N/A')}\n"
                f"**Answer:**\n{answer}\n"
                f"**Citations found:** {len(citations)}\n"
            )

        worker_results_text = "\n---\n".join(worker_text_parts)
        prompt = _SYNTHESIS_PROMPT.format(worker_results=worker_results_text)

        try:
            resp = chat_llm.invoke(
                [SystemMessage(content=prompt)],
                config={
                    "callbacks": callbacks or [],
                    "tags": ["rag_synthesizer"],
                },
            )
            merged_answer = getattr(resp, "content", str(resp))
        except Exception as e:
            logger.warning("RAG synthesis LLM call failed: %s", e)
            # Fallback: concatenate answers
            merged_answer = "\n\n---\n\n".join(
                f"**{r.get('worker_id', 'worker')}:** {r.get('contract', {}).get('answer', '')}"
                for r in results
            )

        # Build the final rendered output
        lines = [merged_answer.strip()]

        if all_citations:
            lines.append("\nCitations:")
            seen = set()
            for c in all_citations:
                cid = c.get("citation_id", "")
                if cid in seen:
                    continue
                seen.add(cid)
                title = c.get("title", "")
                loc = c.get("location", "")
                lines.append(f"- [{cid}] {title} ({loc})")

        if all_warnings:
            lines.append("\nWarnings: " + ", ".join(str(w) for w in all_warnings))

        rendered = "\n".join(lines).strip()

        return {
            "messages": [AIMessage(content=rendered)],
            # Clear rag_results so next parallel run starts fresh
            # (we return [] but the reducer sees this as replacing, not adding)
        }

    return rag_synthesizer_node
