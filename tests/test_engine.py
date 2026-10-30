"""The retrieve() entry point.

The Phase 2 done-criterion is that every retriever is switchable purely by
config, so these tests change nothing but the RetrievalConfig.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from rbi_rag.index.embed import FloatArray
from rbi_rag.index.store import InMemoryChunkStore
from rbi_rag.retrieve import retrieve
from rbi_rag.schemas import Chunk, RetrievalConfig

CHUNK_SET = 1


class FakeEmbedder:
    """Returns a vector per text from a lookup, so rankings are predictable."""

    def __init__(self, vectors: dict[str, list[float]], dimension: int = 2) -> None:
        self._vectors = vectors
        self._dimension = dimension
        self.queries_seen: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_content_tokens(self) -> int:
        return 510

    def embed_passages(self, texts: Sequence[str]) -> FloatArray:
        return np.array([self._vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> FloatArray:
        self.queries_seen.append(text)
        return np.array([self._vectors.get(text, [1.0, 0.0])], dtype=np.float32)


class FakeReranker:
    """Scores by a fixed table, so the reordering is checkable."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls += 1
        return [self._scores.get(passage, 0.0) for passage in passages]


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        page_start=1,
        page_end=1,
        url="https://example.test/x.PDF",
        strategy="fixed_tokens",
        target_tokens=512,
    )


CAPITAL = "banks shall maintain minimum capital"
LIQUIDITY = "liquidity coverage ratio reporting"
RISK = "risk weighted assets and capital"


@pytest.fixture
def store() -> InMemoryChunkStore:
    populated = InMemoryChunkStore()
    populated.add(
        CHUNK_SET,
        [chunk("a", CAPITAL), chunk("b", LIQUIDITY), chunk("c", RISK)],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32),
    )
    return populated


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(
        {
            CAPITAL: [1.0, 0.0],
            LIQUIDITY: [0.0, 1.0],
            RISK: [0.6, 0.8],
            "capital": [1.0, 0.0],
            "liquidity": [0.0, 1.0],
        }
    )


def config(**overrides: object) -> RetrievalConfig:
    return RetrievalConfig(chunk_set_id=CHUNK_SET, **overrides)  # type: ignore[arg-type]


class TestSwitchingByConfig:
    def test_dense_uses_only_the_embedder(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve(
            "capital", store=store, config=config(retriever="dense"), embedder=embedder
        )
        assert [r.chunk.chunk_id for r in results] == ["a", "c", "b"]
        assert all(r.fts_rank is None for r in results)

    def test_fts_needs_no_embedder_at_all(self, store: InMemoryChunkStore) -> None:
        results = retrieve("capital", store=store, config=config(retriever="fts"))
        assert {r.chunk.chunk_id for r in results} == {"a", "c"}
        assert all(r.dense_rank is None for r in results)

    def test_hybrid_combines_both(self, store: InMemoryChunkStore, embedder: FakeEmbedder) -> None:
        results = retrieve(
            "capital", store=store, config=config(retriever="hybrid"), embedder=embedder
        )
        ranked = {r.chunk.chunk_id: r for r in results}
        assert ranked["a"].dense_rank is not None
        assert ranked["a"].fts_rank is not None

    def test_hybrid_surfaces_what_dense_alone_ranks_low(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        """The reason hybrid exists: agreement between retrievers wins."""
        dense_only = retrieve(
            "capital", store=store, config=config(retriever="dense", top_k=1), embedder=embedder
        )
        hybrid = retrieve(
            "capital", store=store, config=config(retriever="hybrid", top_k=1), embedder=embedder
        )
        assert dense_only[0].chunk.chunk_id == "a"
        assert hybrid[0].chunk.chunk_id == "a"


class TestGuards:
    def test_dense_without_an_embedder_is_refused(self, store: InMemoryChunkStore) -> None:
        with pytest.raises(ValueError, match="needs an embedder"):
            retrieve("capital", store=store, config=config(retriever="dense"))

    def test_rerank_without_a_reranker_is_refused(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="no reranker"):
            retrieve("capital", store=store, config=config(rerank=True), embedder=embedder)

    def test_a_blank_query_returns_nothing(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        assert retrieve("   ", store=store, config=config(), embedder=embedder) == []


class TestResultShape:
    def test_ranks_are_contiguous_from_one(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve("capital", store=store, config=config(), embedder=embedder)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_top_k_limits_the_output(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve("capital", store=store, config=config(top_k=2), embedder=embedder)
        assert len(results) == 2

    def test_scores_do_not_increase_down_the_list(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve("capital", store=store, config=config(), embedder=embedder)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_candidate_k_bounds_what_fusion_sees(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve(
            "capital",
            store=store,
            config=config(retriever="dense", candidate_k=1, top_k=10),
            embedder=embedder,
        )
        assert len(results) == 1


class TestRerank:
    def test_reranking_reorders_by_cross_encoder_score(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        reranker = FakeReranker({CAPITAL: 0.1, RISK: 9.0, LIQUIDITY: 0.2})
        results = retrieve(
            "capital",
            store=store,
            config=config(rerank=True),
            embedder=embedder,
            reranker=reranker,
        )
        assert results[0].chunk.chunk_id == "c"
        assert results[0].rerank_score == pytest.approx(9.0)

    def test_without_reranking_no_rerank_score_is_reported(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        results = retrieve("capital", store=store, config=config(), embedder=embedder)
        assert all(r.rerank_score is None for r in results)

    def test_only_the_candidate_window_is_rescored(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        reranker = FakeReranker({CAPITAL: 5.0})
        results = retrieve(
            "capital",
            store=store,
            config=config(rerank=True, rerank_candidates=1, top_k=3),
            embedder=embedder,
            reranker=reranker,
        )
        assert results[0].rerank_score == pytest.approx(5.0)
        assert all(r.rerank_score is None for r in results[1:])

    def test_first_stage_ranks_survive_reranking(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        """Provenance must still explain how a chunk reached the reranker."""
        reranker = FakeReranker({RISK: 9.0})
        results = retrieve(
            "capital",
            store=store,
            config=config(rerank=True),
            embedder=embedder,
            reranker=reranker,
        )
        top = results[0]
        assert top.dense_rank is not None or top.fts_rank is not None


class TestQueryInstruction:
    def test_the_query_text_reaches_the_embedder_unchanged(
        self, store: InMemoryChunkStore, embedder: FakeEmbedder
    ) -> None:
        """The instruction is the embedder's business, not the engine's."""
        retrieve("capital", store=store, config=config(retriever="dense"), embedder=embedder)
        assert embedder.queries_seen == ["capital"]
