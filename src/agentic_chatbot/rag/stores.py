from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List

from agentic_chatbot.config import Settings
from agentic_chatbot.db.chunk_store import ChunkStore
from agentic_chatbot.db.document_store import DocumentStore
from agentic_chatbot.db.memory_store import MemoryStore


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_doc_id(source_type: str, title: str, content_hash: str) -> str:
    """Stable document identifier: TYPE_<10-char sha1 of title+hash>."""
    return f"{source_type.upper()}_{_sha1(title + ':' + content_hash)[:10]}"


@dataclass
class KnowledgeStores:
    """Container for all persistence backends."""
    chunk_store: ChunkStore
    doc_store: DocumentStore
    memory_store: MemoryStore


def load_stores(settings: Settings, embeddings: object) -> KnowledgeStores:
    """Initialise the PostgreSQL connection pool and return KnowledgeStores.

    Applies the schema (idempotent CREATE IF NOT EXISTS) on first call.
    Must be called after load_settings() so that settings.pg_dsn is populated.
    """
    from agentic_chatbot.db.connection import apply_schema, init_pool

    if settings.database_backend != "postgres":
        raise NotImplementedError(
            f"DATABASE_BACKEND={settings.database_backend!r} is not implemented. "
            "Supported backend today: postgres."
        )
    if settings.vector_store_backend != "pgvector":
        raise NotImplementedError(
            f"VECTOR_STORE_BACKEND={settings.vector_store_backend!r} is not implemented. "
            "Supported backend today: pgvector."
        )

    init_pool(settings)
    apply_schema(settings)  # idempotent

    embed_fn: Callable[[str], List[float]] = lambda text: embeddings.embed_query(text)  # type: ignore[attr-defined]

    return KnowledgeStores(
        chunk_store=ChunkStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
        doc_store=DocumentStore(),
        memory_store=MemoryStore(),
    )
