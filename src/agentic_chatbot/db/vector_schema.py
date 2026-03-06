from __future__ import annotations

import re
from typing import Optional

from agentic_chatbot.db.connection import get_conn

_VECTOR_TYPE_RE = re.compile(r"^vector\((\d+)\)$", re.IGNORECASE)


def parse_vector_dimension(type_name: str) -> Optional[int]:
    match = _VECTOR_TYPE_RE.match((type_name or "").strip())
    if not match:
        return None
    return int(match.group(1))


def get_chunks_embedding_dim() -> Optional[int]:
    """Return the configured `chunks.embedding` vector dimension, if discoverable."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'chunks'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            )
            row = cur.fetchone()

    if not row:
        return None
    return parse_vector_dimension(str(row[0]))


def set_chunks_embedding_dim(target_dim: int) -> bool:
    """Alter `chunks.embedding` to vector(target_dim), clearing old vector values.

    Returns True when a schema change was applied, False when already aligned.
    """
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    current_dim = get_chunks_embedding_dim()
    if current_dim == target_dim:
        return False

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
            cur.execute(
                f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({int(target_dim)}) USING NULL"
            )
        conn.commit()
    return True
