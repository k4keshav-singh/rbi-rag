-- Initial schema: documents, chunk sets, chunks with dense and keyword indexes.
--
-- The vector dimension is fixed at 384 to match BAAI/bge-small-en-v1.5. The
-- migration runner asserts that Settings.embedding_dim agrees, because a
-- mismatch would otherwise surface as an opaque insert error much later.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    circular_number TEXT,
    department      TEXT,
    category        TEXT,
    published_on    DATE,
    url             TEXT NOT NULL,
    sha256          TEXT NOT NULL
);

-- One row per chunking configuration. The Phase 4 ablation builds several and
-- compares them, so they have to coexist rather than replace each other.
CREATE TABLE chunk_sets (
    chunk_set_id    SERIAL PRIMARY KEY,
    strategy        TEXT NOT NULL,
    target_tokens   INTEGER NOT NULL CHECK (target_tokens > 0),
    overlap_ratio   REAL NOT NULL CHECK (overlap_ratio >= 0 AND overlap_ratio < 1),
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    git_sha         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (strategy, target_tokens, overlap_ratio, embedding_model)
);

CREATE TABLE chunks (
    chunk_id        TEXT NOT NULL,
    chunk_set_id    INTEGER NOT NULL REFERENCES chunk_sets (chunk_set_id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    page_start      INTEGER NOT NULL CHECK (page_start >= 1),
    page_end        INTEGER NOT NULL CHECK (page_end >= page_start),
    circular_number TEXT,
    url             TEXT NOT NULL,
    section_title   TEXT,
    token_count     INTEGER,
    embedding       vector(384),
    -- Generated so the tsvector can never drift out of step with the text.
    -- 'english' applies Snowball stemming; see DECISIONS.md entry 0015.
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (chunk_set_id, chunk_id)
);

CREATE INDEX chunks_doc_idx ON chunks (doc_id);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);

-- HNSW indexes are created per chunk set by the index builder, as partial
-- indexes. A single index over every set would force post-filtering, which
-- costs recall precisely in the measurement the ablation depends on.
