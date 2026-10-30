"""Storage and search primitives.

``ChunkStore`` is the seam that keeps retrieval testable. The retrievers and the
fusion logic above it never touch psycopg, so their unit tests run against
``InMemoryChunkStore`` with no database and no network. ``PostgresChunkStore``
is exercised separately by tests marked ``integration``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import psycopg

from rbi_rag.index.embed import FloatArray
from rbi_rag.schemas import Chunk, Document

#: Columns needed to rebuild a Chunk, in the order _row_to_chunk expects.
_CHUNK_COLUMNS = """
    c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end,
    c.circular_number, c.url, s.strategy, s.target_tokens,
    c.token_count, c.section_title
"""


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the raw score of whichever retriever produced it."""

    chunk: Chunk
    score: float


class ChunkStore(Protocol):
    """The two searches a retriever needs."""

    def dense_search(
        self, embedding: FloatArray, *, chunk_set_id: int, limit: int, exact: bool = False
    ) -> list[ScoredChunk]: ...

    def fts_search(self, query: str, *, chunk_set_id: int, limit: int) -> list[ScoredChunk]: ...


# --------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------


@contextmanager
def connect(dsn: str) -> Any:
    """Open a connection with the pgvector type adapters registered."""
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(dsn)
    try:
        register_vector(connection)
        yield connection
    finally:
        connection.close()


class PostgresChunkStore:
    """The real store."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    # -- writes ----------------------------------------------------------

    def upsert_documents(self, documents: Iterable[Document]) -> int:
        rows = [
            (
                d.doc_id,
                d.title,
                d.circular_number,
                d.department,
                None,
                d.published_on,
                d.url,
                d.sha256,
            )
            for d in documents
        ]
        if not rows:
            return 0
        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO documents
                    (doc_id, title, circular_number, department, category,
                     published_on, url, sha256)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    circular_number = EXCLUDED.circular_number,
                    department = EXCLUDED.department,
                    published_on = EXCLUDED.published_on,
                    url = EXCLUDED.url,
                    sha256 = EXCLUDED.sha256
                """,
                rows,
            )
        self._connection.commit()
        return len(rows)

    def ensure_chunk_set(
        self,
        *,
        strategy: str,
        target_tokens: int,
        overlap_ratio: float,
        embedding_model: str,
        embedding_dim: int,
        git_sha: str | None = None,
    ) -> int:
        """Return the id of this configuration, creating it if necessary."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chunk_sets
                    (strategy, target_tokens, overlap_ratio, embedding_model,
                     embedding_dim, git_sha)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (strategy, target_tokens, overlap_ratio, embedding_model)
                DO UPDATE SET git_sha = EXCLUDED.git_sha
                RETURNING chunk_set_id
                """,
                (strategy, target_tokens, overlap_ratio, embedding_model, embedding_dim, git_sha),
            )
            row = cursor.fetchone()
        self._connection.commit()
        return int(row[0])

    def replace_chunks(
        self,
        chunk_set_id: int,
        chunks: Sequence[Chunk],
        embeddings: FloatArray,
    ) -> int:
        """Replace every chunk in a set.

        Whole-set replacement rather than incremental update: a chunk id encodes
        its position in the document, so a re-chunk invalidates all of them at
        once and leaving stale rows behind would silently pollute retrieval.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings; refusing to store"
            )
        with self._connection.cursor() as cursor:
            cursor.execute("DELETE FROM chunks WHERE chunk_set_id = %s", (chunk_set_id,))
            if chunks:
                cursor.executemany(
                    """
                    INSERT INTO chunks
                        (chunk_id, chunk_set_id, doc_id, text, page_start, page_end,
                         circular_number, url, section_title, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk_set_id,
                            chunk.doc_id,
                            chunk.text,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.circular_number,
                            chunk.url,
                            chunk.section_title,
                            chunk.token_count,
                            np.asarray(embeddings[position], dtype=np.float32),
                        )
                        for position, chunk in enumerate(chunks)
                    ],
                )
        self._connection.commit()
        return len(chunks)

    def create_hnsw_index(self, chunk_set_id: int, m: int = 16, ef_construction: int = 64) -> str:
        """Create this set's partial HNSW index, if it does not already exist."""
        name = f"chunks_hnsw_set_{chunk_set_id}"
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {name}
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
                WHERE chunk_set_id = {int(chunk_set_id)}
                """
            )
        self._connection.commit()
        return name

    def count_chunks(self, chunk_set_id: int) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM chunks WHERE chunk_set_id = %s", (chunk_set_id,))
            return int(cursor.fetchone()[0])

    # -- searches --------------------------------------------------------

    def dense_search(
        self, embedding: FloatArray, *, chunk_set_id: int, limit: int, exact: bool = False
    ) -> list[ScoredChunk]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._connection.cursor() as cursor:
            if exact:
                # Force a sequential scan so results are exact kNN. HNSW is
                # approximate, and evaluation must not attribute its recall
                # loss to the retriever under test.
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute("SET LOCAL enable_bitmapscan = off")
            cursor.execute(
                f"""
                SELECT {_CHUNK_COLUMNS}, 1 - (c.embedding <=> %s) AS score
                FROM chunks c
                JOIN chunk_sets s USING (chunk_set_id)
                WHERE c.chunk_set_id = %s AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vector, chunk_set_id, vector, limit),
            )
            return [_row_to_scored(row) for row in cursor.fetchall()]

    def fts_search(self, query: str, *, chunk_set_id: int, limit: int) -> list[ScoredChunk]:
        """Keyword search over one chunk set.

        The query is a **disjunction** of its own lexemes, not
        ``websearch_to_tsquery``. That function is conjunctive: the question
        "what minimum capital ratio must banks maintain" becomes
        ``minimum & capit & ratio & must & bank & maintain``, and a passage
        saying exactly the right thing still fails because it lacks "must".
        Measured on the fixture corpus, that returned nothing for a question
        whose answer was sitting in the index, which would have been reported
        as a finding about keyword retrieval rather than the bug it was.

        Lexical retrieval is disjunctive and ranked -- BM25 works this way too
        -- so terms are OR-ed and ``ts_rank_cd`` discriminates. The lexemes
        come from Postgres's own tokenizer, so nothing user-supplied is
        interpolated. A query of nothing but stopwords yields a NULL tsquery
        and therefore no rows, rather than a syntax error.

        The cost: explicit operators (quoted phrases, negation) are not
        honoured here. This retriever serves natural-language questions, where
        conjunctive matching is by far the greater loss.
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH q AS (
                    SELECT to_tsquery(
                        'english',
                        NULLIF(
                            array_to_string(
                                ARRAY(
                                    SELECT lexeme
                                    FROM unnest(to_tsvector('english', %s))
                                ),
                                ' | '
                            ),
                            ''
                        )
                    ) AS query
                )
                SELECT {_CHUNK_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
                FROM chunks c
                JOIN chunk_sets s USING (chunk_set_id),
                     q
                WHERE c.chunk_set_id = %s AND c.tsv @@ q.query
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, chunk_set_id, limit),
            )
            return [_row_to_scored(row) for row in cursor.fetchall()]


def _row_to_scored(row: Sequence[Any]) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=row[0],
            doc_id=row[1],
            text=row[2],
            page_start=row[3],
            page_end=row[4],
            circular_number=row[5],
            url=row[6],
            strategy=row[7],
            target_tokens=row[8],
            token_count=row[9],
            section_title=row[10],
        ),
        score=float(row[11]),
    )


# --------------------------------------------------------------------------
# In-memory
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class InMemoryChunkStore:
    """A store that holds everything in memory, for unit tests.

    Dense search is exact cosine similarity, so it matches Postgres. Keyword
    search is a term-overlap score, which is *not* ``ts_rank_cd`` -- it exists
    to give fusion and ranking logic something deterministic to work against.
    Anything that depends on Postgres ranking specifically belongs in an
    integration test.

    Both stores match **disjunctively**. They did not always: this fake OR-ed
    its terms while Postgres AND-ed them, so unit tests passed on questions
    that returned nothing against the real database. Keep the two agreeing on
    semantics, or the fake stops being evidence about anything.
    """

    def __init__(self) -> None:
        self._chunks: dict[int, list[Chunk]] = {}
        self._embeddings: dict[int, FloatArray] = {}

    def add(
        self, chunk_set_id: int, chunks: Sequence[Chunk], embeddings: FloatArray | None = None
    ) -> None:
        self._chunks[chunk_set_id] = list(chunks)
        if embeddings is None:
            embeddings = np.zeros((len(chunks), 1), dtype=np.float32)
        if len(embeddings) != len(chunks):
            raise ValueError("chunks and embeddings must be the same length")
        self._embeddings[chunk_set_id] = np.asarray(embeddings, dtype=np.float32)

    def dense_search(
        self, embedding: FloatArray, *, chunk_set_id: int, limit: int, exact: bool = False
    ) -> list[ScoredChunk]:
        chunks = self._chunks.get(chunk_set_id, [])
        if not chunks:
            return []
        matrix = self._embeddings[chunk_set_id]
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        scores = matrix @ query
        order = np.argsort(-scores)[:limit]
        return [ScoredChunk(chunk=chunks[i], score=float(scores[i])) for i in order]

    def fts_search(self, query: str, *, chunk_set_id: int, limit: int) -> list[ScoredChunk]:
        terms = set(_tokenise(query))
        if not terms:
            return []
        scored: list[ScoredChunk] = []
        for chunk in self._chunks.get(chunk_set_id, []):
            tokens = _tokenise(chunk.text)
            if not tokens:
                continue
            hits = sum(1 for token in tokens if token in terms)
            if hits:
                scored.append(ScoredChunk(chunk=chunk, score=hits / len(tokens)))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:limit]
