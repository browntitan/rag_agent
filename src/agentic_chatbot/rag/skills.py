"""Skills loader — reads Markdown prompt files at runtime.

Skills files live in data/skills/:
  skills.md              – shared preamble injected into ALL agents
  general_agent.md       – GeneralAgent-specific instructions
  rag_agent.md           – RAGAgent-specific instructions
  supervisor_agent.md    – Supervisor routing instructions
  utility_agent.md       – Utility agent instructions

If a file is missing, each function falls back to a hard-coded default.
Edit the .md files to change agent behaviour without touching Python code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from agentic_chatbot.config import Settings

# ---------------------------------------------------------------------------
# Defaults (used when the corresponding .md file is absent)
# ---------------------------------------------------------------------------

_DEFAULT_SHARED = ""

_DEFAULT_GENERAL_SYSTEM = (
    "You are an agentic chatbot that can call tools to solve the user's request.\n"
    "Your priorities are: (1) correct tool selection, (2) correct tool arguments, "
    "(3) clear synthesis of tool results.\n\n"
    "Operating rules:\n"
    "- When a task requires tools, use them. When it doesn't, answer directly.\n"
    "- If multiple tools are needed, create a short numbered PLAN, then execute step-by-step.\n"
    "- Use rag_agent_tool for questions about the KB, uploaded documents, policies, "
    "contracts, requirements, runbooks, or anything that needs citations.\n"
    "- The rag_agent_tool returns JSON: {answer, citations, used_citation_ids, confidence}. "
    "Present the 'answer' field to the user with citations. Do NOT dump raw JSON.\n"
    "- For multi-document tasks (compare, diff, find requirements), call rag_agent_tool with "
    "a detailed query describing the full task. Pass preferred_doc_ids_csv to constrain scope.\n"
    "- Use scratchpad_context_key to pass previous RAG findings as context into the next call.\n"
    "- Use memory_save to persist important facts the user has confirmed across sessions.\n"
    "- If tool output is insufficient, explain what is missing and ask a follow-up question.\n"
    "- Keep the final answer concise and user-friendly.\n"
)

_DEFAULT_RAG_SYSTEM = (
    "You are a specialist RAG (Retrieval-Augmented Generation) agent.\n\n"
    "Your job is to answer the user's QUERY using ONLY evidence retrieved from the indexed documents.\n\n"
    "Operating rules:\n"
    "1. ALWAYS call resolve_document first if the user refers to a document by name or description.\n"
    "2. Call list_document_structure to understand a document's clause/section outline before "
    "extracting specific clauses.\n"
    "3. Use search_document for targeted single-doc search; search_all_documents for broad search.\n"
    "4. Use extract_clauses when the user references specific numbered clauses (e.g. 'Clause 33').\n"
    "5. Use extract_requirements when asked to 'find all requirements', 'list all shall statements', "
    "or similar extraction tasks.\n"
    "6. For cross-document comparisons:\n"
    "   - Use diff_documents to get the structural outline diff first.\n"
    "   - Then use compare_clauses for specific clause-by-clause analysis.\n"
    "   - Store partial results with scratchpad_write between steps.\n"
    "7. When processing multiple documents sequentially (e.g. 'look at doc_1 then doc_2'), "
    "resolve each document separately, process them in order, and use scratchpad_write to "
    "accumulate findings before synthesising a final answer.\n"
    "8. If evidence is insufficient after multiple searches, clearly state what was not found.\n"
    "9. NEVER fabricate document content — only report what the tools return.\n"
    "10. Cite inline using (chunk_id) values from the tool results.\n"
)

_DEFAULT_SUPERVISOR_SYSTEM = (
    "You are a supervisor agent that coordinates specialist agents.\n"
    "Route to: rag_agent (document questions), utility_agent (calculations, memory, listing docs), "
    "parallel_rag (multi-document comparison), or __end__ (simple greetings / direct answers).\n\n"
    'Respond with JSON: {"reasoning": "...", "next_agent": "...", "direct_answer": "", "rag_sub_tasks": []}\n'
)

_DEFAULT_UTILITY_SYSTEM = (
    "You are a utility agent that handles calculations, document listing, and persistent memory.\n"
    "Tools: calculator, list_indexed_docs, memory_save, memory_load, memory_list.\n"
    "Always use the calculator for math. Always call memory_load to recall facts.\n"
)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_md(path: Path) -> Optional[str]:
    """Read a markdown file. Returns stripped content or None if missing/empty."""
    try:
        content = path.read_text(encoding="utf-8").strip()
        return content if content else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def load_shared_skills(settings: Settings) -> str:
    """Load data/skills/skills.md. Returns empty string if file is absent."""
    path = settings.data_dir / "skills" / "skills.md"
    return _load_md(path) or _DEFAULT_SHARED


def load_general_agent_skills(settings: Settings) -> str:
    """Return the GeneralAgent system prompt.

    Combines:
      1. Shared preamble from data/skills/skills.md
      2. Agent-specific instructions from data/skills/general_agent.md

    Falls back to hard-coded defaults if files are missing.
    """
    shared = load_shared_skills(settings)
    specific = _load_md(settings.data_dir / "skills" / "general_agent.md")
    body = specific or _DEFAULT_GENERAL_SYSTEM
    if shared:
        return f"{shared}\n\n---\n\n{body}".strip()
    return body.strip()


def load_rag_agent_skills(settings: Settings) -> str:
    """Return the RAGAgent system prompt.

    Combines:
      1. Shared preamble from data/skills/skills.md
      2. Agent-specific instructions from data/skills/rag_agent.md

    Falls back to hard-coded defaults if files are missing.
    """
    shared = load_shared_skills(settings)
    specific = _load_md(settings.data_dir / "skills" / "rag_agent.md")
    body = specific or _DEFAULT_RAG_SYSTEM
    if shared:
        return f"{shared}\n\n---\n\n{body}".strip()
    return body.strip()


def load_supervisor_skills(settings: Settings) -> str:
    """Return the Supervisor agent system prompt.

    Combines:
      1. Shared preamble from data/skills/skills.md
      2. Supervisor-specific instructions from data/skills/supervisor_agent.md

    Falls back to hard-coded defaults if files are missing.
    """
    shared = load_shared_skills(settings)
    specific = _load_md(settings.data_dir / "skills" / "supervisor_agent.md")
    body = specific or _DEFAULT_SUPERVISOR_SYSTEM
    if shared:
        return f"{shared}\n\n---\n\n{body}".strip()
    return body.strip()


def load_utility_agent_skills(settings: Settings) -> str:
    """Return the Utility agent system prompt.

    Combines:
      1. Shared preamble from data/skills/skills.md
      2. Utility-specific instructions from data/skills/utility_agent.md

    Falls back to hard-coded defaults if files are missing.
    """
    shared = load_shared_skills(settings)
    specific = _load_md(settings.data_dir / "skills" / "utility_agent.md")
    body = specific or _DEFAULT_UTILITY_SYSTEM
    if shared:
        return f"{shared}\n\n---\n\n{body}".strip()
    return body.strip()
