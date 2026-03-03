-- ============================================================
-- Agentic RAG Chatbot — PostgreSQL Schema
-- Run once: psql -d ragdb -f schema.sql
-- Requires: pgvector >= 0.5 (HNSW), pg_trgm
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ------------------------------------------------------------
-- documents: one row per ingested file
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id             TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    title              TEXT NOT NULL,
    source_type        TEXT NOT NULL,           -- 'kb' | 'upload'
    source_path        TEXT,
    content_hash       TEXT NOT NULL,
    num_chunks         INTEGER DEFAULT 0,
    ingested_at        TIMESTAMPTZ DEFAULT now(),
    file_type          TEXT,                    -- 'pdf' | 'txt' | 'md' | 'docx'
    doc_structure_type TEXT DEFAULT 'general'   -- see chunk_type values below
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE documents SET tenant_id = 'local-dev' WHERE tenant_id IS NULL OR tenant_id = '';
ALTER TABLE documents ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS documents_tenant_idx
    ON documents(tenant_id);

CREATE INDEX IF NOT EXISTS documents_tenant_source_idx
    ON documents(tenant_id, source_type);

-- ------------------------------------------------------------
-- chunks: one row per document chunk
--
-- embedding dimension is set at CREATE time; must match EMBEDDING_DIM env var.
-- To change dimensions you must DROP and recreate this table.
--
-- chunk_type values:
--   'general'     – plain prose (no detected structure)
--   'clause'      – numbered clause / article
--   'section'     – section heading block
--   'requirement' – contains shall/must/REQ-NNN language
--   'header'      – document title / heading only
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    chunk_index    INTEGER NOT NULL,
    page_number    INTEGER,
    clause_number  TEXT,          -- e.g. '3', '3.2', '10.1.4'
    section_title  TEXT,          -- heading text extracted from the boundary line
    content        TEXT NOT NULL,
    embedding      vector(768),   -- CHANGE 768 if EMBEDDING_DIM != 768
    ts             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    chunk_type     TEXT DEFAULT 'general'
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE chunks c
SET tenant_id = COALESCE((SELECT d.tenant_id FROM documents d WHERE d.doc_id = c.doc_id), 'local-dev')
WHERE c.tenant_id IS NULL OR c.tenant_id = '';
ALTER TABLE chunks ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE chunks ALTER COLUMN tenant_id SET NOT NULL;

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx
    ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_doc_idx
    ON chunks(tenant_id, doc_id);

-- HNSW can be created on an empty table (unlike IVFFlat)
-- m=16, ef_construction=64 are conservative defaults; tune for your dataset size
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_ts_gin_idx
    ON chunks USING GIN(ts);

CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx
    ON chunks(chunk_type);

CREATE INDEX IF NOT EXISTS chunks_clause_number_idx
    ON chunks(tenant_id, doc_id, clause_number)
    WHERE clause_number IS NOT NULL;

-- ------------------------------------------------------------
-- memory: persistent cross-turn key-value store per tenant+session
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory (
    id          SERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'local-dev',
    session_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE memory ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE memory SET tenant_id = 'local-dev' WHERE tenant_id IS NULL OR tenant_id = '';
ALTER TABLE memory ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE memory ALTER COLUMN tenant_id SET NOT NULL;

-- Keep old uniqueness for backward compatibility if it exists, and add the
-- tenant-aware unique index required by ON CONFLICT (tenant_id, session_id, key).
CREATE UNIQUE INDEX IF NOT EXISTS memory_tenant_session_key_uniq
    ON memory(tenant_id, session_id, key);

CREATE INDEX IF NOT EXISTS memory_tenant_session_idx
    ON memory(tenant_id, session_id);
