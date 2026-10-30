"""Adapters between the measurement code and the live system.

The evaluation modules take callables. This is where those callables are built
from a real store, embedder, reranker and LLM, and where responses are cached
so a re-run of an unchanged arm costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rbi_rag.config import Settings
from rbi_rag.eval.ablation import ArmSpec
from rbi_rag.eval.cache import ResponseCache, cache_key
from rbi_rag.eval.judge import Judge
from rbi_rag.generate.client import LLMClient
from rbi_rag.generate.pipeline import AnswerOutcome, answer_question
from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import ChunkStore
from rbi_rag.retrieve.engine import retrieve as retrieve_chunks
from rbi_rag.retrieve.rerank import Reranker
from rbi_rag.schemas import EvalItem, RetrievalConfig, RetrievedChunk


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
