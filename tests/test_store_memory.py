"""The in-memory store that lets retrieval be unit-tested without Postgres."""

import numpy as np
import pytest

from rbi_rag.index.store import InMemoryChunkStore
from rbi_rag.schemas import Chunk


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        text=text,
        page_start=1,
        page_end=1,
        url="https://example.test/x.PDF",
        strategy="fixed_tokens",
        target_tokens=512,
    )


@pytest.fixture
def store() -> InMemoryChunkStore:
    populated = InMemoryChunkStore()
    populated.add(
        1,
        [
            chunk("a", "banks shall maintain a minimum capital adequacy ratio"),
            chunk("b", "the reporting of liquidity coverage is quarterly"),
            chunk("c", "capital adequacy is computed on risk weighted assets"),
        ],
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
            dtype=np.float32,
        ),
    )
    return populated


class TestDenseSearch:
    def test_ranks_by_cosine_similarity(self, store: InMemoryChunkStore) -> None:
        results = store.dense_search(
            np.array([1.0, 0.0], dtype=np.float32), chunk_set_id=1, limit=3
        )
        assert [r.chunk.chunk_id for r in results] == ["a", "c", "b"]

    def test_scores_are_the_dot_product(self, store: InMemoryChunkStore) -> None:
        results = store.dense_search(
            np.array([1.0, 0.0], dtype=np.float32), chunk_set_id=1, limit=1
        )
        assert results[0].score == pytest.approx(1.0)

    def test_respects_the_limit(self, store: InMemoryChunkStore) -> None:
        results = store.dense_search(
            np.array([1.0, 0.0], dtype=np.float32), chunk_set_id=1, limit=2
        )
        assert len(results) == 2

    def test_an_unknown_chunk_set_is_empty(self, store: InMemoryChunkStore) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        assert store.dense_search(query, chunk_set_id=9, limit=3) == []


class TestFtsSearch:
    def test_finds_chunks_sharing_terms(self, store: InMemoryChunkStore) -> None:
        results = store.fts_search("capital adequacy", chunk_set_id=1, limit=5)
        assert {r.chunk.chunk_id for r in results} == {"a", "c"}

    def test_excludes_chunks_with_no_overlap(self, store: InMemoryChunkStore) -> None:
        results = store.fts_search("capital", chunk_set_id=1, limit=5)
        assert "b" not in {r.chunk.chunk_id for r in results}

    def test_an_empty_query_matches_nothing(self, store: InMemoryChunkStore) -> None:
        assert store.fts_search("   ", chunk_set_id=1, limit=5) == []

    def test_ordering_is_deterministic(self, store: InMemoryChunkStore) -> None:
        first = store.fts_search("capital adequacy", chunk_set_id=1, limit=5)
        second = store.fts_search("capital adequacy", chunk_set_id=1, limit=5)
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]


class TestAdd:
    def test_rejects_mismatched_embeddings(self) -> None:
        populated = InMemoryChunkStore()
        with pytest.raises(ValueError, match="same length"):
            populated.add(1, [chunk("a", "x")], np.zeros((2, 2), dtype=np.float32))
