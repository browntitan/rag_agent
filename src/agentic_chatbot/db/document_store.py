from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2.extras

from agentic_chatbot.db.connection import get_conn


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_type: str                    # 'kb' | 'upload'
    content_hash: str
    source_path: str = ""
    num_chunks: int = 0
    ingested_at: str = ""
    file_type: str = ""
    doc_structure_type: str = "general"


class DocumentStore:
    """CRUD operations against the `documents` table."""

    def upsert_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record (keyed on doc_id)."""
        ingested_at = doc.ingested_at or (dt.datetime.utcnow().isoformat() + "Z")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (doc_id, title, source_type, source_path, content_hash,
                         num_chunks, ingested_at, file_type, doc_structure_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title              = EXCLUDED.title,
                        source_type        = EXCLUDED.source_type,
                        source_path        = EXCLUDED.source_path,
                        content_hash       = EXCLUDED.content_hash,
                        num_chunks         = EXCLUDED.num_chunks,
                        ingested_at        = EXCLUDED.ingested_at,
                        file_type          = EXCLUDED.file_type,
                        doc_structure_type = EXCLUDED.doc_structure_type
                    """,
                    (
                        doc.doc_id,
                        doc.title,
                        doc.source_type,
                        doc.source_path,
                        doc.content_hash,
                        doc.num_chunks,
                        ingested_at,
                        doc.file_type,
                        doc.doc_structure_type,
                    ),
                )
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Return a DocumentRecord by doc_id, or None if not found."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE doc_id = %s",
                    (doc_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(dict(row))

    def document_exists(self, doc_id: str, content_hash: str) -> bool:
        """Return True if a document with this doc_id and content_hash is already stored."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE doc_id = %s AND content_hash = %s",
                    (doc_id, content_hash),
                )
                return cur.fetchone() is not None

    def list_documents(self, source_type: str = "") -> List[DocumentRecord]:
        """Return all documents, optionally filtered by source_type ('kb' or 'upload')."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if source_type:
                    cur.execute(
                        "SELECT * FROM documents WHERE source_type = %s ORDER BY ingested_at",
                        (source_type,),
                    )
                else:
                    cur.execute("SELECT * FROM documents ORDER BY ingested_at")
                rows = cur.fetchall()
        return [_row_to_record(dict(r)) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        """Delete a document record (chunks cascade via FK)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            conn.commit()

    def fuzzy_search_title(self, hint: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return documents whose title fuzzy-matches hint, ranked by similarity.

        Uses PostgreSQL pg_trgm similarity(). Requires the pg_trgm extension.
        Returns list of dicts: {doc_id, title, source_type, doc_structure_type, score}.
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT doc_id, title, source_type, doc_structure_type,
                           similarity(lower(title), lower(%s)) AS score
                    FROM documents
                    WHERE similarity(lower(title), lower(%s)) > 0.1
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (hint, hint, limit),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_all_titles(self) -> List[Dict[str, str]]:
        """Return [{doc_id, title}] for all documents — used by resolve_document tool."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT doc_id, title, source_type FROM documents ORDER BY title")
                rows = cur.fetchall()
        return [dict(r) for r in rows]


def _row_to_record(row: Dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row.get("doc_id", ""),
        title=row.get("title", ""),
        source_type=row.get("source_type", ""),
        content_hash=row.get("content_hash", ""),
        source_path=row.get("source_path") or "",
        num_chunks=row.get("num_chunks") or 0,
        ingested_at=str(row.get("ingested_at") or ""),
        file_type=row.get("file_type") or "",
        doc_structure_type=row.get("doc_structure_type") or "general",
    )
