from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn


@dataclass
class SkillPackRecord:
    skill_id: str
    name: str
    agent_scope: str
    checksum: str
    tenant_id: str = "local-dev"
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    version: str = "1"
    enabled: bool = True
    source_path: str = ""
    description: str = ""
    updated_at: str = ""


@dataclass
class SkillChunkMatch:
    skill_id: str
    name: str
    agent_scope: str
    content: str
    chunk_index: int
    score: float
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)


class SkillStore:
    """Persistence for retrievable skill packs and their vectorized chunks."""

    def __init__(self, embed_fn: Callable[[str], List[float]], embedding_dim: int = 768):
        self._embed = embed_fn
        self.embedding_dim = embedding_dim

    def upsert_skill_pack(self, record: SkillPackRecord, chunks: List[str]) -> None:
        timestamp = record.updated_at or (dt.datetime.utcnow().isoformat() + "Z")
        rows: List[Tuple[Any, ...]] = []
        for index, content in enumerate(chunks):
            rows.append((
                f"{record.skill_id}#chunk{index:04d}",
                record.skill_id,
                record.tenant_id,
                index,
                content,
                self._embed(content),
            ))

        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO skills
                        (skill_id, tenant_id, name, agent_scope, tool_tags, task_tags,
                         version, enabled, source_path, checksum, description, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (skill_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        name = EXCLUDED.name,
                        agent_scope = EXCLUDED.agent_scope,
                        tool_tags = EXCLUDED.tool_tags,
                        task_tags = EXCLUDED.task_tags,
                        version = EXCLUDED.version,
                        enabled = EXCLUDED.enabled,
                        source_path = EXCLUDED.source_path,
                        checksum = EXCLUDED.checksum,
                        description = EXCLUDED.description,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.skill_id,
                        record.tenant_id,
                        record.name,
                        record.agent_scope,
                        record.tool_tags,
                        record.task_tags,
                        record.version,
                        record.enabled,
                        record.source_path,
                        record.checksum,
                        record.description,
                        timestamp,
                    ),
                )
                cur.execute(
                    "DELETE FROM skill_chunks WHERE tenant_id = %s AND skill_id = %s",
                    (record.tenant_id, record.skill_id),
                )
                if rows:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO skill_chunks
                            (skill_chunk_id, skill_id, tenant_id, chunk_index, content, embedding)
                        VALUES %s
                        ON CONFLICT (skill_chunk_id) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            chunk_index = EXCLUDED.chunk_index,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                        template="(%s, %s, %s, %s, %s, %s::vector)",
                    )
            conn.commit()

    def list_skill_packs(
        self,
        *,
        tenant_id: str = "local-dev",
        agent_scope: str = "",
        enabled_only: bool = False,
    ) -> List[SkillPackRecord]:
        sql = "SELECT * FROM skills WHERE tenant_id = %s"
        params: List[Any] = [tenant_id]
        if agent_scope:
            sql += " AND agent_scope = %s"
            params.append(agent_scope)
        if enabled_only:
            sql += " AND enabled = TRUE"
        sql += " ORDER BY agent_scope, name"

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [_row_to_skill_pack(dict(row)) for row in rows]

    def get_skill_pack(self, skill_id: str, *, tenant_id: str = "local-dev") -> Optional[SkillPackRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM skills WHERE tenant_id = %s AND skill_id = %s LIMIT 1",
                    (tenant_id, skill_id),
                )
                row = cur.fetchone()
        return _row_to_skill_pack(dict(row)) if row else None

    def get_skill_chunks(self, skill_id: str, *, tenant_id: str = "local-dev") -> List[Dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT skill_chunk_id, skill_id, chunk_index, content
                    FROM skill_chunks
                    WHERE tenant_id = %s AND skill_id = %s
                    ORDER BY chunk_index
                    """,
                    (tenant_id, skill_id),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def delete_skill_pack(self, skill_id: str, *, tenant_id: str = "local-dev") -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM skills WHERE tenant_id = %s AND skill_id = %s", (tenant_id, skill_id))
            conn.commit()

    def vector_search(
        self,
        query: str,
        *,
        tenant_id: str = "local-dev",
        top_k: int = 4,
        agent_scope: str = "",
        tool_tags: Optional[List[str]] = None,
        task_tags: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[SkillChunkMatch]:
        embedding = self._embed(query)
        tool_tags = [tag for tag in (tool_tags or []) if tag]
        task_tags = [tag for tag in (task_tags or []) if tag]

        sql = """
            SELECT sc.skill_id,
                   s.name,
                   s.agent_scope,
                   s.tool_tags,
                   s.task_tags,
                   sc.content,
                   sc.chunk_index,
                   1 - (sc.embedding <=> %s::vector) AS score
            FROM skill_chunks sc
            JOIN skills s ON s.skill_id = sc.skill_id AND s.tenant_id = sc.tenant_id
            WHERE sc.tenant_id = %s
              AND s.tenant_id = %s
        """
        params: List[Any] = [embedding, tenant_id, tenant_id]
        if enabled_only:
            sql += " AND s.enabled = TRUE"
        if agent_scope:
            sql += " AND s.agent_scope = %s"
            params.append(agent_scope)
        if tool_tags:
            sql += " AND s.tool_tags && %s"
            params.append(tool_tags)
        if task_tags:
            sql += " AND s.task_tags && %s"
            params.append(task_tags)
        sql += " ORDER BY sc.embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, top_k])

        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            SkillChunkMatch(
                skill_id=row["skill_id"],
                name=row["name"],
                agent_scope=row["agent_scope"],
                content=row["content"],
                chunk_index=int(row["chunk_index"]),
                score=float(row["score"]),
                tool_tags=list(row.get("tool_tags") or []),
                task_tags=list(row.get("task_tags") or []),
            )
            for row in rows
        ]


def _row_to_skill_pack(row: Dict[str, Any]) -> SkillPackRecord:
    return SkillPackRecord(
        skill_id=row.get("skill_id", ""),
        tenant_id=row.get("tenant_id") or "local-dev",
        name=row.get("name", ""),
        agent_scope=row.get("agent_scope", ""),
        tool_tags=list(row.get("tool_tags") or []),
        task_tags=list(row.get("task_tags") or []),
        version=row.get("version") or "1",
        enabled=bool(row.get("enabled", True)),
        source_path=row.get("source_path") or "",
        checksum=row.get("checksum") or "",
        description=row.get("description") or "",
        updated_at=str(row.get("updated_at") or ""),
    )
