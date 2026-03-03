from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agentic_chatbot.config import Settings

DEFAULT_JUDGE_GRADING_PROMPT = (
    "You are a retrieval relevance grader.\n"
    "Given a QUESTION and a list of CHUNKS, assign each chunk a relevance score:\n"
    "3 = directly answers the question or contains key required facts\n"
    "2 = partially relevant / useful supporting information\n"
    "1 = tangentially related\n"
    "0 = not relevant\n\n"
    "Return ONLY valid JSON in this exact schema:\n"
    "{\"grades\": [{\"chunk_id\": \"...\", \"relevance\": 0, \"reason\": \"...\"}, ...]}\n\n"
    "QUESTION: {{QUESTION}}\n\n"
    "CHUNKS: {{CHUNKS_JSON}}"
)


DEFAULT_JUDGE_REWRITE_PROMPT = (
    "You are a query rewriting assistant for retrieval.\n"
    "Rewrite the QUESTION into a better search query.\n"
    "- Prefer concrete nouns and key terms\n"
    "- Include synonyms if helpful\n"
    "- Remove filler words\n"
    "Return ONLY valid JSON: {\"rewritten_query\": \"...\"}.\n\n"
    "ATTEMPT: {{ATTEMPT}}\n"
    "QUESTION: {{QUESTION}}\n"
    "CONVERSATION_CONTEXT: {{CONVERSATION_CONTEXT}}\n"
)


DEFAULT_GROUNDED_ANSWER_PROMPT = (
    "You are a grounded QA assistant.\n"
    "Answer the QUESTION using ONLY the EVIDENCE snippets provided.\n"
    "Rules:\n"
    "- If a claim depends on evidence, cite it inline using (citation_id).\n"
    "- If evidence is insufficient, say what is missing and ask a clarifying question.\n"
    "- Do NOT fabricate document details.\n\n"
    "Return ONLY valid JSON in this schema:\n"
    "{\"answer\": \"...\", \"used_citation_ids\": [\"\"], \"followups\": [\"\"], \"warnings\": [\"\"], \"confidence_hint\": 0.0}\n\n"
    "QUESTION: {{QUESTION}}\n"
    "CONVERSATION_CONTEXT: {{CONVERSATION_CONTEXT}}\n\n"
    "EVIDENCE: {{EVIDENCE_JSON}}"
)


DEFAULT_RAG_SYNTHESIS_PROMPT = (
    "Based on all the tool results above, produce your final answer.\n"
    "Return ONLY valid JSON in this exact schema:\n"
    "{\"answer\": \"...\", \"used_citation_ids\": [\"...\"], "
    "\"followups\": [\"...\"], \"warnings\": [\"...\"], \"confidence_hint\": 0.0}\n\n"
    "Rules:\n"
    "- answer: comprehensive, cite inline using (citation_id) from chunk_ids you retrieved.\n"
    "- used_citation_ids: list of chunk_ids actually cited in the answer.\n"
    "- followups: 2-3 suggested next questions.\n"
    "- warnings: list any missing information or uncertainty.\n"
    "- confidence_hint: float 0.0-1.0 reflecting your confidence.\n"
    "\nOriginal query: {{ORIGINAL_QUERY}}"
)


DEFAULT_PARALLEL_RAG_SYNTHESIS_PROMPT = (
    "You are merging results from multiple parallel document searches.\n\n"
    "Below are the individual results from each worker. Each worker searched a different\n"
    "document or scope. Your job is to:\n\n"
    "1. Combine the findings into a single coherent answer\n"
    "2. Preserve ALL citations from every worker (use inline (chunk_id) references)\n"
    "3. Highlight differences and similarities between documents\n"
    "4. Note any gaps or asymmetries (one doc has content the other doesn't)\n"
    "5. Include warnings from any worker\n\n"
    "{{WORKER_RESULTS}}\n\n"
    "Produce a clear, structured response that addresses the user's original question.\n"
    "Use headings or bullet points to organise cross-document comparisons."
)


def _read_text_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text if text else None
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _load_local_or_default(path: Path, default: str) -> str:
    return _read_text_file(path) or default


def _ensure_local_backend(backend: str, kind: str) -> None:
    if backend != "local":
        raise NotImplementedError(
            f"{kind} backend {backend!r} is not implemented yet. "
            "Set backend to 'local' for now."
        )


def render_template(template: str, values: Dict[str, Any]) -> str:
    text = template
    for key, value in values.items():
        token = "{{" + str(key) + "}}"
        if isinstance(value, (dict, list)):
            repl = json.dumps(value, ensure_ascii=False)
        else:
            repl = str(value)
        text = text.replace(token, repl)
    return text


def load_judge_grading_prompt(settings: Settings) -> str:
    _ensure_local_backend(settings.prompts_backend, "PROMPTS")
    return _load_local_or_default(settings.judge_grading_prompt_path, DEFAULT_JUDGE_GRADING_PROMPT)


def load_judge_rewrite_prompt(settings: Settings) -> str:
    _ensure_local_backend(settings.prompts_backend, "PROMPTS")
    return _load_local_or_default(settings.judge_rewrite_prompt_path, DEFAULT_JUDGE_REWRITE_PROMPT)


def load_grounded_answer_prompt(settings: Settings) -> str:
    _ensure_local_backend(settings.prompts_backend, "PROMPTS")
    return _load_local_or_default(settings.grounded_answer_prompt_path, DEFAULT_GROUNDED_ANSWER_PROMPT)


def load_rag_synthesis_prompt(settings: Settings) -> str:
    _ensure_local_backend(settings.prompts_backend, "PROMPTS")
    return _load_local_or_default(settings.rag_synthesis_prompt_path, DEFAULT_RAG_SYNTHESIS_PROMPT)


def load_parallel_rag_synthesis_prompt(settings: Settings) -> str:
    _ensure_local_backend(settings.prompts_backend, "PROMPTS")
    return _load_local_or_default(settings.parallel_rag_synthesis_prompt_path, DEFAULT_PARALLEL_RAG_SYNTHESIS_PROMPT)
