"""One-time migration utility: Chroma + JSON docstore → PostgreSQL.

Usage (from CLI):
    python run.py migrate

This is safe to run multiple times — existing records are skipped via
ON CONFLICT DO NOTHING / document_exists() checks.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict


def migrate_docstore_to_pg(docstore_path: pathlib.Path) -> int:
    """Read the legacy JSON docstore and insert rows into the documents table.

    Returns the number of documents inserted.
    """
    from agentic_chatbot.db.document_store import DocumentRecord, DocumentStore

    if not docstore_path.exists():
        print(f"  [migration] No legacy docstore found at {docstore_path} — skipping.")
        return 0

    raw: Dict[str, Any] = json.loads(docstore_path.read_text(encoding="utf-8"))
    store = DocumentStore()
    inserted = 0

    for doc_id, meta in raw.items():
        if store.document_exists(doc_id, meta.get("content_hash", "")):
            continue
        record = DocumentRecord(
            doc_id=doc_id,
            title=meta.get("title", ""),
            source_type=meta.get("source_type", "kb"),
            content_hash=meta.get("content_hash", ""),
            source_path=meta.get("source_path", ""),
            num_chunks=meta.get("num_chunks", 0),
            ingested_at=meta.get("ingested_at", ""),
            file_type=pathlib.Path(meta.get("source_path", "")).suffix.lstrip("."),
            doc_structure_type="general",
        )
        store.upsert_document(record)
        inserted += 1

    print(f"  [migration] Inserted {inserted} documents from legacy docstore.")
    return inserted


def migrate_chroma_to_pg(
    chroma_dir: pathlib.Path,
    embed_fn: Any,
    embedding_dim: int = 768,
) -> int:
    """Extract chunks from an existing Chroma collection and insert into PostgreSQL.

    Embeddings stored in Chroma are extracted directly to avoid re-embedding.
    Returns the number of chunks inserted.
    """
    from agentic_chatbot.db.chunk_store import ChunkRecord, ChunkStore

    try:
        import chromadb  # type: ignore
    except ImportError:
        print("  [migration] chromadb not installed — cannot migrate Chroma data.")
        return 0

    if not chroma_dir.exists():
        print(f"  [migration] No Chroma dir at {chroma_dir} — skipping.")
        return 0

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection("agentic_chatbot")
    except Exception:
        print("  [migration] Chroma collection 'agentic_chatbot' not found — skipping.")
        return 0

    total = collection.count()
    if total == 0:
        print("  [migration] Chroma collection is empty — nothing to migrate.")
        return 0

    chunk_store = ChunkStore(embed_fn=embed_fn, embedding_dim=embedding_dim)
    batch_size = 200
    offset = 0
    inserted = 0

    while offset < total:
        result = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=batch_size,
            offset=offset,
        )
        records: list[ChunkRecord] = []
        for text, meta, emb in zip(
            result.get("documents", []),
            result.get("metadatas", []),
            result.get("embeddings", []),
        ):
            meta = meta or {}
            records.append(
                ChunkRecord(
                    chunk_id=str(meta.get("chunk_id", "")),
                    doc_id=str(meta.get("doc_id", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    content=text or "",
                    chunk_type=str(meta.get("chunk_type", "general")),
                    page_number=meta.get("page"),
                    clause_number=meta.get("clause_number"),
                    section_title=meta.get("section_title"),
                    embedding=list(emb) if emb is not None else None,
                )
            )
        if records:
            chunk_store.add_chunks(records)
            inserted += len(records)

        offset += batch_size

    print(f"  [migration] Migrated {inserted} chunks from Chroma.")
    return inserted
