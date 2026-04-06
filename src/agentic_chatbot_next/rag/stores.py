from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.persistence.postgres.chunks import ChunkStore
from agentic_chatbot_next.persistence.postgres.connection import apply_schema, init_pool
from agentic_chatbot_next.persistence.postgres.documents import DocumentStore
from agentic_chatbot_next.persistence.postgres.skills import SkillStore


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_doc_id(source_type: str, title: str, content_hash: str, tenant_id: str) -> str:
    key = f"{tenant_id}:{title}:{content_hash}"
    return f"{source_type.upper()}_{_sha1(key)[:10]}"


@dataclass
class KnowledgeStores:
    chunk_store: ChunkStore
    doc_store: DocumentStore
    memory_store: object | None
    skill_store: SkillStore


def load_stores(settings: Settings, embeddings: object) -> KnowledgeStores:
    if settings.database_backend != "postgres":
        raise NotImplementedError(
            f"DATABASE_BACKEND={settings.database_backend!r} is not implemented. Supported: postgres."
        )
    if settings.vector_store_backend != "pgvector":
        raise NotImplementedError(
            f"VECTOR_STORE_BACKEND={settings.vector_store_backend!r} is not implemented. Supported: pgvector."
        )

    init_pool(settings)
    apply_schema(settings)
    embed_fn: Callable[[str], List[float]] = lambda text: embeddings.embed_query(text)  # type: ignore[attr-defined]
    return KnowledgeStores(
        chunk_store=ChunkStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
        doc_store=DocumentStore(),
        memory_store=None,
        skill_store=SkillStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
    )
