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
    collection_id      TEXT NOT NULL DEFAULT 'default',
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
ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection_id TEXT;
UPDATE documents SET collection_id = 'default' WHERE collection_id IS NULL OR collection_id = '';
ALTER TABLE documents ALTER COLUMN collection_id SET DEFAULT 'default';
ALTER TABLE documents ALTER COLUMN collection_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS documents_tenant_idx
    ON documents(tenant_id);

CREATE INDEX IF NOT EXISTS documents_tenant_source_idx
    ON documents(tenant_id, source_type);

CREATE INDEX IF NOT EXISTS documents_tenant_collection_idx
    ON documents(tenant_id, collection_id);

-- ------------------------------------------------------------
-- chunks: one row per document chunk
--
-- embedding dimension is injected from Settings.EMBEDDING_DIM when schema is applied.
-- Existing databases may still require `python run.py migrate-embedding-dim --yes`
-- to realign and reindex.
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
    collection_id  TEXT NOT NULL DEFAULT 'default',
    chunk_index    INTEGER NOT NULL,
    page_number    INTEGER,
    clause_number  TEXT,          -- e.g. '3', '3.2', '10.1.4'
    section_title  TEXT,          -- heading text extracted from the boundary line
    content        TEXT NOT NULL,
    embedding      vector(__EMBEDDING_DIM__),
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
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id TEXT;
UPDATE chunks c
SET collection_id = COALESCE((SELECT d.collection_id FROM documents d WHERE d.doc_id = c.doc_id), 'default')
WHERE c.collection_id IS NULL OR c.collection_id = '';
ALTER TABLE chunks ALTER COLUMN collection_id SET DEFAULT 'default';
ALTER TABLE chunks ALTER COLUMN collection_id SET NOT NULL;

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx
    ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_doc_idx
    ON chunks(tenant_id, doc_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_collection_idx
    ON chunks(tenant_id, collection_id);

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
-- skills: skill-pack metadata indexed separately from KB docs
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills (
    skill_id      TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'local-dev',
    name          TEXT NOT NULL,
    agent_scope   TEXT NOT NULL,
    tool_tags     TEXT[] DEFAULT '{}'::TEXT[],
    task_tags     TEXT[] DEFAULT '{}'::TEXT[],
    version       TEXT NOT NULL DEFAULT '1',
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    source_path   TEXT,
    checksum      TEXT NOT NULL,
    description   TEXT DEFAULT '',
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS skills_tenant_scope_idx
    ON skills(tenant_id, agent_scope);

CREATE INDEX IF NOT EXISTS skills_enabled_idx
    ON skills(tenant_id, enabled);

-- ------------------------------------------------------------
-- skill_chunks: retrieval surface for skill-pack chunks
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_chunks (
    skill_chunk_id TEXT PRIMARY KEY,
    skill_id       TEXT NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    chunk_index    INTEGER NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(__EMBEDDING_DIM__)
);

CREATE INDEX IF NOT EXISTS skill_chunks_skill_idx
    ON skill_chunks(skill_id);

CREATE INDEX IF NOT EXISTS skill_chunks_tenant_idx
    ON skill_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS skill_chunks_embedding_hnsw_idx
    ON skill_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

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
