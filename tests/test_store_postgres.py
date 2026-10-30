"""Postgres-backed storage and search.

Marked ``integration``: needs a running database. Start it with
``uv run poe db-up`` and run with ``uv run poe test-integration``.

These cover what the in-memory store deliberately cannot: real pgvector
distance, real ``ts_rank_cd`` ranking, the generated tsvector column, and
whether exact search and the HNSW index actually agree.
"""

from collections.abc import Iterator

import numpy as np
import psycopg
import pytest

from rbi_rag.config import get_settings
from rbi_rag.index.migrate import migrate
from rbi_rag.index.store import PostgresChunkStore
from rbi_rag.schemas import Chunk, Document

pytestmark = pytest.mark.integration

DIM = 384


def unit_vector(*leading: float) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for index, value in enumerate(leading):
        vector[index] = value
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


@pytest.fixture
def connection() -> Iterator[psycopg.Connection]:
    from pgvector.psycopg import register_vector

    settings = get_settings()
    try:
        conn = psycopg.connect(settings.postgres_dsn, connect_timeout=5)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres is not reachable ({exc}); run `uv run poe db-up`")

    register_vector(conn)
    migrate(conn)
    # Test rows are tagged with a test-only embedding_model, which is part of
    # the chunk_sets unique key, so a real index is never touched.
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM chunk_sets WHERE embedding_model LIKE 'test-%'")
        cursor.execute("DELETE FROM documents WHERE doc_id LIKE 'test-%'")
    conn.commit()
    try:
        yield conn
    finally:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM chunk_sets WHERE embedding_model LIKE 'test-%'")
            cursor.execute("DELETE FROM documents WHERE doc_id LIKE 'test-%'")
        conn.commit()
        conn.close()


@pytest.fixture
def store(connection: psycopg.Connection) -> PostgresChunkStore:
    populated = PostgresChunkStore(connection)
    populated.upsert_documents(
        [
            Document(
                doc_id="test-1",
                title="Test Direction",
                circular_number="RBI/DOR/2023-24/106",
                url="https://example.test/x.PDF",
                sha256="a" * 64,
            )
        ]
    )
    return populated


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="test-1",
        text=text,
        page_start=1,
        page_end=2,
        circular_number="RBI/DOR/2023-24/106",
        url="https://example.test/x.PDF",
        strategy="fixed_tokens",
        target_tokens=512,
        token_count=9,
        section_title="3. Capital",
    )


CAPITAL = "banks shall maintain a minimum capital adequacy ratio of nine per cent"
LIQUIDITY = "the liquidity coverage ratio is reported every quarter without fail"
RISK = "capital is computed on risk weighted assets under these directions"


@pytest.fixture
def populated(store: PostgresChunkStore) -> int:
    chunk_set_id = store.ensure_chunk_set(
        strategy="fixed_tokens",
        target_tokens=512,
        overlap_ratio=0.15,
        embedding_model="test-model",
        embedding_dim=DIM,
    )
    store.replace_chunks(
        chunk_set_id,
        [chunk("a", CAPITAL), chunk("b", LIQUIDITY), chunk("c", RISK)],
        np.vstack([unit_vector(1.0), unit_vector(0.0, 1.0), unit_vector(0.6, 0.8)]),
    )
    return chunk_set_id


class TestMigrations:
    def test_running_twice_applies_nothing_the_second_time(
        self, connection: psycopg.Connection
    ) -> None:
        assert migrate(connection) == []

    def test_the_tsvector_column_is_generated(self, connection: psycopg.Connection) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT is_generated FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'tsv'
                """
            )
            assert cursor.fetchone()[0] == "ALWAYS"


class TestChunkSets:
    def test_is_idempotent(self, store: PostgresChunkStore) -> None:
        kwargs = {
            "strategy": "fixed_tokens",
            "target_tokens": 512,
            "overlap_ratio": 0.15,
            "embedding_model": "test-model",
            "embedding_dim": DIM,
        }
        first = store.ensure_chunk_set(**kwargs)  # type: ignore[arg-type]
        assert store.ensure_chunk_set(**kwargs) == first  # type: ignore[arg-type]

    def test_different_configurations_coexist(self, store: PostgresChunkStore) -> None:
        """The ablation compares them side by side, so they must not collide."""
        base = {
            "strategy": "fixed_tokens",
            "overlap_ratio": 0.15,
            "embedding_model": "test-model",
            "embedding_dim": DIM,
        }
        small = store.ensure_chunk_set(target_tokens=256, **base)  # type: ignore[arg-type]
        large = store.ensure_chunk_set(target_tokens=512, **base)  # type: ignore[arg-type]
        assert small != large


class TestReplaceChunks:
    def test_stores_and_counts(self, store: PostgresChunkStore, populated: int) -> None:
        assert store.count_chunks(populated) == 3

    def test_replacing_removes_the_previous_rows(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """Stale chunks would silently pollute every later retrieval."""
        store.replace_chunks(populated, [chunk("z", CAPITAL)], np.vstack([unit_vector(1.0)]))
        assert store.count_chunks(populated) == 1

    def test_refuses_a_length_mismatch(self, store: PostgresChunkStore, populated: int) -> None:
        with pytest.raises(ValueError, match="refusing to store"):
            store.replace_chunks(
                populated, [chunk("a", CAPITAL)], np.zeros((0, DIM), dtype=np.float32)
            )


class TestDenseSearch:
    def test_ranks_by_cosine_distance(self, store: PostgresChunkStore, populated: int) -> None:
        results = store.dense_search(unit_vector(1.0), chunk_set_id=populated, limit=3)
        assert [r.chunk.chunk_id for r in results] == ["a", "c", "b"]

    def test_score_is_cosine_similarity(self, store: PostgresChunkStore, populated: int) -> None:
        results = store.dense_search(unit_vector(1.0), chunk_set_id=populated, limit=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_rebuilds_the_full_chunk(self, store: PostgresChunkStore, populated: int) -> None:
        """Citations need the page range and circular number to survive storage."""
        top = store.dense_search(unit_vector(1.0), chunk_set_id=populated, limit=1)[0].chunk
        assert top.page_start == 1
        assert top.page_end == 2
        assert top.circular_number == "RBI/DOR/2023-24/106"
        assert top.section_title == "3. Capital"
        assert top.strategy == "fixed_tokens"
        assert top.target_tokens == 512

    def test_exact_search_agrees_with_the_index(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """If these ever diverge, recall numbers measured through HNSW are suspect."""
        store.create_hnsw_index(populated)
        approximate = store.dense_search(unit_vector(1.0), chunk_set_id=populated, limit=3)
        exact = store.dense_search(unit_vector(1.0), chunk_set_id=populated, limit=3, exact=True)
        assert [r.chunk.chunk_id for r in approximate] == [r.chunk.chunk_id for r in exact]

    def test_other_chunk_sets_are_not_searched(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        other = store.ensure_chunk_set(
            strategy="fixed_tokens",
            target_tokens=256,
            overlap_ratio=0.15,
            embedding_model="test-model",
            embedding_dim=DIM,
        )
        assert store.dense_search(unit_vector(1.0), chunk_set_id=other, limit=3) == []


class TestFtsSearch:
    def test_finds_matching_chunks(self, store: PostgresChunkStore, populated: int) -> None:
        results = store.fts_search("capital adequacy", chunk_set_id=populated, limit=5)
        assert "a" in {r.chunk.chunk_id for r in results}

    def test_stemming_matches_inflected_forms(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """'reported' in the text must be found by 'reporting'."""
        results = store.fts_search("reporting", chunk_set_id=populated, limit=5)
        assert "b" in {r.chunk.chunk_id for r in results}

    def test_non_matching_queries_return_nothing(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        assert store.fts_search("zygomorphic", chunk_set_id=populated, limit=5) == []

    def test_a_natural_language_question_matches(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """Regression: conjunctive matching returned nothing for questions.

        websearch_to_tsquery builds `minimum & capit & ratio & must & bank &
        maintain`, and the passage that answers this does not contain "must",
        so every arm scored zero. Lexical retrieval has to be disjunctive.
        """
        results = store.fts_search(
            "what minimum capital ratio must banks maintain",
            chunk_set_id=populated,
            limit=5,
        )
        assert results, "a question whose answer is indexed must match something"
        assert results[0].chunk.chunk_id == "a"

    def test_a_query_of_only_stopwords_returns_nothing_rather_than_erroring(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """An empty lexeme list would be a to_tsquery syntax error."""
        assert store.fts_search("the and of", chunk_set_id=populated, limit=5) == []

    def test_search_operators_are_not_honoured_but_do_not_crash(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """The documented cost of going disjunctive."""
        assert store.fts_search('"capital adequacy" -liquidity', chunk_set_id=populated, limit=5)

    def test_ranking_prefers_the_denser_match(
        self, store: PostgresChunkStore, populated: int
    ) -> None:
        """OR-matching without good ranking would be useless."""
        results = store.fts_search("capital adequacy ratio", chunk_set_id=populated, limit=5)
        assert results[0].chunk.chunk_id == "a"

    def test_scores_are_descending(self, store: PostgresChunkStore, populated: int) -> None:
        results = store.fts_search("capital", chunk_set_id=populated, limit=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestHnswIndex:
    def test_creating_it_twice_is_safe(self, store: PostgresChunkStore, populated: int) -> None:
        name = store.create_hnsw_index(populated)
        assert store.create_hnsw_index(populated) == name

    def test_the_index_is_partial_to_this_chunk_set(
        self, store: PostgresChunkStore, populated: int, connection: psycopg.Connection
    ) -> None:
        """A shared index would force post-filtering and cost recall."""
        name = store.create_hnsw_index(populated)
        with connection.cursor() as cursor:
            cursor.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (name,))
            definition = cursor.fetchone()[0]
        assert "hnsw" in definition.lower()
        assert f"chunk_set_id = {populated}" in definition
