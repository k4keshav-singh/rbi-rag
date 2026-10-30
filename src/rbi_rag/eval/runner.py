"""Adapters between the measurement code and the live system.

The evaluation modules take callables. This is where those callables are built
from a real store, embedder, reranker and LLM, and where responses are cached
so a re-run of an unchanged arm costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from rbi_rag.config import Settings
from rbi_rag.eval.ablation import ArmSpec, evaluate_retrieval
from rbi_rag.eval.cache import ResponseCache, cache_key
from rbi_rag.eval.gate import GATE_RETRIEVERS
from rbi_rag.eval.judge import Judge
from rbi_rag.eval.snapshot import PrecomputedEmbedder, Snapshot
from rbi_rag.generate.client import LLMClient
from rbi_rag.generate.pipeline import AnswerOutcome, answer_question
from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import ChunkStore, PostgresChunkStore
from rbi_rag.retrieve.engine import retrieve as retrieve_chunks
from rbi_rag.retrieve.rerank import Reranker
from rbi_rag.schemas import ChunkStrategy, EvalItem, RetrievalConfig, RetrievedChunk, RetrieverName


def config_for(spec: ArmSpec, settings: Settings, top_k: int) -> RetrievalConfig:
    """The retrieval configuration for one arm.

    ``exact=True`` always: HNSW is approximate, and letting its recall loss
    into a comparison between retrievers would make it indistinguishable from
    a real difference between them.
    """
    return RetrievalConfig(
        retriever=spec.retriever,
        chunk_set_id=spec.chunk_set_id,
        top_k=top_k,
        candidate_k=settings.candidate_k,
        rrf_k=settings.rrf_k,
        rerank=spec.rerank,
        rerank_candidates=settings.rerank_candidates,
        use_query_instruction=settings.use_query_instruction,
        exact=True,
    )


def make_retrieve(
    store: ChunkStore,
    config: RetrievalConfig,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> Callable[[str], Sequence[RetrievedChunk]]:
    """A question -> ranked chunks callable for one arm."""

    def run(question: str) -> Sequence[RetrievedChunk]:
        return retrieve_chunks(
            question, store=store, config=config, embedder=embedder, reranker=reranker
        )

    return run


def make_answer(
    settings: Settings,
    store: ChunkStore,
    llm: LLMClient,
    config: RetrievalConfig,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> Callable[[EvalItem], AnswerOutcome]:
    """An item -> AnswerOutcome callable for one arm."""

    def run(item: EvalItem) -> AnswerOutcome:
        return answer_question(
            item.question,
            settings=settings,
            store=store,
            llm=llm,
            config=config,
            embedder=embedder,
            reranker=reranker,
        )

    return run


def make_judge_faithful(
    judge: Judge, cache: ResponseCache, arm_id: str
) -> Callable[[EvalItem, AnswerOutcome], bool | None]:
    """An (item, outcome) -> faithful? callable, cached by content.

    The key covers the answer and every cited quote, so an arm that produced a
    different answer gets judged again rather than reusing a verdict about
    different text.
    """

    def run(item: EvalItem, outcome: AnswerOutcome) -> bool | None:
        if not outcome.citations:
            # Nothing was cited, so there is nothing to check support against.
            return None
        key = cache_key(
            kind="faithfulness",
            arm=arm_id,
            item=item.id,
            question=item.question,
            answer=outcome.answer,
            citations=[(c.chunk_id, c.quote) for c in outcome.citations],
        )
        cached = cache.get(key)
        if cached is not None and "faithful" in cached:
            return bool(cached["faithful"])

        result = judge.judge(item.question, outcome.answer, outcome.citations)
        cache.put(
            key,
            {
                "faithful": result.verdict.faithful,
                "unsupported_claims": result.verdict.unsupported_claims,
                "model": result.model,
            },
        )
        return result.verdict.faithful

    return run


# -- the CI snapshot and gate ------------------------------------------------


def load_snapshot_into_postgres(
    settings: Settings, connection: Any, loaded: Snapshot
) -> tuple[PostgresChunkStore, int]:
    """Put the committed snapshot into the database under its own chunk set.

    The chunk set is tagged with a snapshot-only embedding_model, which is part
    of the chunk_sets unique key, so it can never collide with or overwrite a
    real index on the same machine.
    """
    store = PostgresChunkStore(connection)
    store.upsert_documents(loaded.documents())
    chunk_set_id = store.ensure_chunk_set(
        strategy=loaded.manifest.strategy,
        target_tokens=loaded.manifest.target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
        embedding_model=loaded.manifest.chunk_set_model,
        embedding_dim=loaded.manifest.embedding_dim,
    )
    store.replace_chunks(chunk_set_id, loaded.chunks, loaded.embeddings)
    return store, chunk_set_id


def measure_snapshot(
    settings: Settings, connection: Any, loaded: Snapshot
) -> dict[str, dict[str, float | None]]:
    """Measure every gated arm against the snapshot.

    Queries are embedded by replaying the committed vectors, so this needs no
    model and no network -- but the retrieval path above the embedder is the
    production one, running against real Postgres. The in-memory store would
    have been faster and would have measured the wrong thing: its keyword
    search is term overlap, not ts_rank_cd.
    """
    store, chunk_set_id = load_snapshot_into_postgres(settings, connection, loaded)
    embedder = PrecomputedEmbedder(
        [item.question for item in loaded.questions],
        loaded.query_vectors,
        loaded.manifest.embedding_model,
    )

    measured: dict[str, dict[str, float | None]] = {}
    for retriever in GATE_RETRIEVERS:
        spec = ArmSpec(
            strategy=cast("ChunkStrategy", loaded.manifest.strategy),
            target_tokens=loaded.manifest.target_tokens,
            retriever=cast("RetrieverName", retriever),
            rerank=False,
            chunk_set_id=chunk_set_id,
        )
        config = config_for(spec, settings, max(settings.eval_k_values))
        measured[spec.arm_id] = evaluate_retrieval(
            loaded.questions,
            make_retrieve(store, config, embedder, None),
            settings.eval_k_values,
        )
    return measured


def chunk_set_config(connection: Any, chunk_set_id: int) -> tuple[str, int]:
    """The strategy and chunk size behind a chunk set id."""
    from rbi_rag.index.build import list_chunk_sets

    for row in list_chunk_sets(connection):
        if int(row["chunk_set_id"]) == chunk_set_id:
            return str(row["strategy"]), int(row["target_tokens"])
    raise ValueError(f"no chunk set {chunk_set_id}")
