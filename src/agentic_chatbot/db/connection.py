from __future__ import annotations

import contextlib
from typing import Generator

import psycopg2
from psycopg2 import pool as pg_pool

from agentic_chatbot.config import Settings

_pool: pg_pool.ThreadedConnectionPool | None = None


def init_pool(settings: Settings, minconn: int = 1, maxconn: int = 10) -> None:
    """Initialise the singleton connection pool. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        return
    _pool = pg_pool.ThreadedConnectionPool(
        minconn=minconn,
        maxconn=maxconn,
        dsn=settings.pg_dsn,
    )


def get_pool() -> pg_pool.ThreadedConnectionPool:
    if _pool is None:
        raise RuntimeError(
            "PostgreSQL connection pool is not initialised. "
            "Call db.connection.init_pool(settings) at startup."
        )
    return _pool


@contextlib.contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that borrows a connection from the pool and returns it on exit."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def apply_schema(settings: Settings, schema_path: str | None = None) -> None:
    """Apply schema.sql to the target database.

    Idempotent — all CREATE statements use IF NOT EXISTS.
    Call once at startup or from the CLI migrate command.
    """
    import pathlib

    if schema_path is None:
        schema_path = str(pathlib.Path(__file__).parent / "schema.sql")

    sql_template = pathlib.Path(schema_path).read_text(encoding="utf-8")
    sql = sql_template.replace("__EMBEDDING_DIM__", str(int(settings.embedding_dim)))
    init_pool(settings)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def close_pool() -> None:
    """Close all connections. Call on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
