from __future__ import annotations

from typing import Dict, List, Optional

import psycopg2.extras

from agentic_chatbot.db.connection import get_conn


class MemoryStore:
    """Persistent key-value memory store per session, backed by the `memory` table."""

    def save(self, session_id: str, key: str, value: str) -> None:
        """Upsert a key-value pair for a session."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory (session_id, key, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id, key) DO UPDATE SET
                        value      = EXCLUDED.value,
                        updated_at = now()
                    """,
                    (session_id, key, value),
                )
            conn.commit()

    def get(self, session_id: str, key: str) -> Optional[str]:
        """Return the stored value for a key, or None."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM memory WHERE session_id = %s AND key = %s",
                    (session_id, key),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def load(self, session_id: str) -> Dict[str, str]:
        """Return all key-value pairs for a session as a dict."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT key, value FROM memory WHERE session_id = %s ORDER BY updated_at",
                    (session_id,),
                )
                rows = cur.fetchall()
        return {r["key"]: r["value"] for r in rows}

    def list_keys(self, session_id: str) -> List[str]:
        """Return all stored keys for a session, ordered by last update."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT key FROM memory WHERE session_id = %s ORDER BY updated_at DESC",
                    (session_id,),
                )
                rows = cur.fetchall()
        return [r[0] for r in rows]

    def delete(self, session_id: str, key: str) -> None:
        """Delete a specific key."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory WHERE session_id = %s AND key = %s",
                    (session_id, key),
                )
            conn.commit()

    def clear_session(self, session_id: str) -> None:
        """Delete all memory for a session."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory WHERE session_id = %s", (session_id,))
            conn.commit()
