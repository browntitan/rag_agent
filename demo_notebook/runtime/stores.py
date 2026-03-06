from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from .config import NotebookSettings


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    source_type: str
    content_hash: str
    num_chunks: int
    ingested_at: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    embedding: List[float]
    clause_number: Optional[str] = None
    section_title: Optional[str] = None


class PostgresVectorStore:
    """Standalone pgvector-backed store for the notebook demo product."""

    def __init__(self, settings: NotebookSettings, embeddings: Any):
        self.settings = settings
        self.embeddings = embeddings

    def _connect(self):
        return psycopg2.connect(self.settings.pg_dsn)

    def ensure_schema(self) -> None:
        dim = int(self.settings.embedding_dim)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dn_documents (
                        doc_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        num_chunks INTEGER NOT NULL DEFAULT 0,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS dn_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL REFERENCES dn_documents(doc_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        clause_number TEXT,
                        section_title TEXT,
                        content TEXT NOT NULL,
                        embedding vector({dim}),
                        ts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                    )
                    """
                )

                cur.execute(
                    "CREATE INDEX IF NOT EXISTS dn_chunks_doc_idx ON dn_chunks(doc_id, chunk_index)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS dn_chunks_ts_idx ON dn_chunks USING GIN(ts)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS dn_chunks_embedding_idx ON dn_chunks USING hnsw (embedding vector_cosine_ops)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS dn_docs_title_trgm_idx ON dn_documents USING gin (title gin_trgm_ops)"
                )

                # Validate vector dimension if table already existed with a prior shape.
                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = 'dn_chunks' AND a.attname = 'embedding'
                    """
                )
                row = cur.fetchone()
                if row and row[0] != f"vector({dim})":
                    raise RuntimeError(
                        f"dn_chunks.embedding uses {row[0]} but NOTEBOOK_EMBEDDING_DIM={dim}. "
                        "Drop dn_chunks and dn_documents to reinitialize notebook schema."
                    )
            conn.commit()

    def list_documents(self) -> List[DocumentRecord]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM dn_documents ORDER BY title")
                rows = cur.fetchall()
        return [
            DocumentRecord(
                doc_id=r["doc_id"],
                title=r["title"],
                source_path=r["source_path"],
                source_type=r["source_type"],
                content_hash=r["content_hash"],
                num_chunks=r["num_chunks"],
                ingested_at=str(r["ingested_at"]),
            )
            for r in rows
        ]

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dn_documents (doc_id, title, source_path, source_type, content_hash, num_chunks, ingested_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_path = EXCLUDED.source_path,
                        source_type = EXCLUDED.source_type,
                        content_hash = EXCLUDED.content_hash,
                        num_chunks = EXCLUDED.num_chunks,
                        ingested_at = EXCLUDED.ingested_at
                    """,
                    (
                        record.doc_id,
                        record.title,
                        record.source_path,
                        record.source_type,
                        record.content_hash,
                        record.num_chunks,
                        record.ingested_at,
                    ),
                )
            conn.commit()

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM dn_documents WHERE doc_id=%s", (doc_id,))
            conn.commit()

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        expected_dim = int(self.settings.embedding_dim)
        observed_dims = sorted({len(c.embedding or []) for c in chunks})
        if observed_dims != [expected_dim]:
            observed_str = ", ".join(str(d) for d in observed_dims) if observed_dims else "unknown"
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"NOTEBOOK_EMBEDDING_DIM={expected_dim} but provider returned dimension(s) [{observed_str}]. "
                "Set NOTEBOOK_EMBEDDING_DIM to match your embeddings model "
                "(for Ollama nomic-embed-text this is typically 768), then drop and recreate "
                "dn_chunks/dn_documents before reindexing."
            )

        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.chunk_index,
                c.clause_number,
                c.section_title,
                c.content,
                c.embedding,
            )
            for c in chunks
        ]

        with self._connect() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO dn_chunks (chunk_id, doc_id, chunk_index, clause_number, section_title, content, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        clause_number = EXCLUDED.clause_number,
                        section_title = EXCLUDED.section_title
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s, %s, %s::vector)",
                )
            conn.commit()

    def _embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)

    def vector_search(self, query: str, *, top_k: int = 8, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        vector = self._embed_query(query)
        with self._connect() as conn:
            register_vector(conn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if doc_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.clause_number, c.section_title,
                               d.title,
                               1 - (c.embedding <=> %s::vector) AS score
                        FROM dn_chunks c
                        JOIN dn_documents d ON d.doc_id = c.doc_id
                        WHERE c.doc_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector, doc_id, vector, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.clause_number, c.section_title,
                               d.title,
                               1 - (c.embedding <=> %s::vector) AS score
                        FROM dn_chunks c
                        JOIN dn_documents d ON d.doc_id = c.doc_id
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector, vector, top_k),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def keyword_search(self, query: str, *, top_k: int = 8, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if doc_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.clause_number, c.section_title,
                               d.title,
                               ts_rank_cd(c.ts, plainto_tsquery('english', %s)) AS score
                        FROM dn_chunks c
                        JOIN dn_documents d ON d.doc_id = c.doc_id
                        WHERE c.doc_id = %s AND c.ts @@ plainto_tsquery('english', %s)
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, doc_id, query, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.clause_number, c.section_title,
                               d.title,
                               ts_rank_cd(c.ts, plainto_tsquery('english', %s)) AS score
                        FROM dn_chunks c
                        JOIN dn_documents d ON d.doc_id = c.doc_id
                        WHERE c.ts @@ plainto_tsquery('english', %s)
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, query, top_k),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def hybrid_search(self, query: str, *, top_k_vector: int = 8, top_k_keyword: int = 8, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        vector_rows = self.vector_search(query, top_k=top_k_vector, doc_id=doc_id)
        keyword_rows = self.keyword_search(query, top_k=top_k_keyword, doc_id=doc_id)

        merged: Dict[str, Dict[str, Any]] = {}
        for idx, row in enumerate(vector_rows):
            chunk_id = row["chunk_id"]
            merged.setdefault(chunk_id, dict(row))
            merged[chunk_id]["hybrid_score"] = merged[chunk_id].get("hybrid_score", 0.0) + (1.0 / (1 + idx))
            merged[chunk_id]["methods"] = sorted(set(merged[chunk_id].get("methods", []) + ["vector"]))

        for idx, row in enumerate(keyword_rows):
            chunk_id = row["chunk_id"]
            merged.setdefault(chunk_id, dict(row))
            merged[chunk_id]["hybrid_score"] = merged[chunk_id].get("hybrid_score", 0.0) + (1.0 / (1 + idx))
            merged[chunk_id]["methods"] = sorted(set(merged[chunk_id].get("methods", []) + ["keyword"]))

        ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return ranked

    def search_titles(self, title_hint: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT doc_id, title, source_type,
                           similarity(lower(title), lower(%s)) AS score
                    FROM dn_documents
                    WHERE similarity(lower(title), lower(%s)) > 0.05
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (title_hint, title_hint, limit),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_document_content(self, doc_id: str) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM dn_chunks WHERE doc_id=%s ORDER BY chunk_index",
                    (doc_id,),
                )
                rows = cur.fetchall()
        return "\n".join(r[0] for r in rows)

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.clause_number, c.section_title, d.title
                    FROM dn_chunks c
                    JOIN dn_documents d ON d.doc_id = c.doc_id
                    WHERE c.doc_id=%s
                    ORDER BY c.chunk_index
                    """,
                    (doc_id,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]



def make_doc_id(path_str: str, content_hash: str) -> str:
    digest = hashlib.sha1(f"{path_str}:{content_hash}".encode("utf-8")).hexdigest()[:12]
    return f"DN_{digest}"


def utcnow_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def json_dumps_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
