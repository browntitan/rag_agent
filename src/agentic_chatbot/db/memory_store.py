from __future__ import annotations

from typing import Dict, List, Optional

import psycopg2.extras

from agentic_chatbot.db.connection import get_conn


class MemoryStore:
    """Persistent key-value memory store per tenant/session, backed by `memory` table."""

    def save(self, tenant_id: str, session_id: str, key: str, value: str) -> None:
        """Upsert a key-value pair for a tenant session."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory (tenant_id, session_id, key, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (tenant_id, session_id, key) DO UPDATE SET
                        value      = EXCLUDED.value,
                        updated_at = now()
                    """,
                    (tenant_id, session_id, key, value),
                )
            conn.commit()

    def get(self, tenant_id: str, session_id: str, key: str) -> Optional[str]:
        """Return the stored value for a key, or None."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM memory WHERE tenant_id = %s AND session_id = %s AND key = %s",
                    (tenant_id, session_id, key),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def load(self, tenant_id: str, session_id: str) -> Dict[str, str]:
        """Return all key-value pairs for a session as a dict."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT key, value FROM memory WHERE tenant_id = %s AND session_id = %s ORDER BY updated_at",
                    (tenant_id, session_id),
                )
                rows = cur.fetchall()
        return {r["key"]: r["value"] for r in rows}

    def list_keys(self, tenant_id: str, session_id: str) -> List[str]:
        """Return all stored keys for a session, ordered by last update."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key FROM memory WHERE tenant_id = %s AND session_id = %s ORDER BY updated_at DESC",
                    (tenant_id, session_id),
                )
                rows = cur.fetchall()
        return [r[0] for r in rows]

    def delete(self, tenant_id: str, session_id: str, key: str) -> None:
        """Delete a specific key."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory WHERE tenant_id = %s AND session_id = %s AND key = %s",
                    (tenant_id, session_id, key),
                )
            conn.commit()

    def clear_session(self, tenant_id: str, session_id: str) -> None:
        """Delete all memory for a session."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory WHERE tenant_id = %s AND session_id = %s",
                    (tenant_id, session_id),
                )
            conn.commit()
