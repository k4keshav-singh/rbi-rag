"""The single retrieval entry point.

Every variant the ablation compares -- dense, keyword, hybrid, each with or
without reranking -- goes through ``retrieve``. Which one runs is decided
entirely by ``RetrievalConfig``, so an ablation arm is a config object and
nothing else.

The returned ``RetrievedChunk`` carries the per-retriever ranks alongside the
final score, so a result can be explained rather than just trusted.
"""

from __future__ import annotations

from collections.abc import Sequence

from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import ChunkStore, ScoredChunk
from rbi_rag.retrieve.fuse import fuse_and_order, rank_positions
from rbi_rag.retrieve.rerank import Reranker
from rbi_rag.schemas import Chunk, RetrievalConfig, RetrievedChunk


def retrieve(
    query: str,
    *,
    store: ChunkStore,
    config: RetrievalConfig,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top chunks for ``query`` under ``config``."""
    if not query.strip():
        return []
    if config.rerank and reranker is None:
        raise ValueError("reranking was requested but no reranker was supplied")

    dense_hits: list[ScoredChunk] = []
    fts_hits: list[ScoredChunk] = []

    if config.retriever in ("dense", "hybrid"):
        if embedder is None:
            raise ValueError(f"the {config.retriever} retriever needs an embedder")
        embedding = embedder.embed_query(query)[0]
        dense_hits = store.dense_search(
            embedding,
            chunk_set_id=config.chunk_set_id,
            limit=config.candidate_k,
            exact=config.exact,
        )

    if config.retriever in ("fts", "hybrid"):
        fts_hits = store.fts_search(
            query, chunk_set_id=config.chunk_set_id, limit=config.candidate_k
        )

    by_id: dict[str, Chunk] = {hit.chunk.chunk_id: hit.chunk for hit in (*dense_hits, *fts_hits)}
    dense_ranks = rank_positions([hit.chunk.chunk_id for hit in dense_hits])
    fts_ranks = rank_positions([hit.chunk.chunk_id for hit in fts_hits])

    ordered = _order(config, dense_hits, fts_hits)

    if config.rerank and reranker is not None:
        ordered = _rerank(query, ordered, by_id, reranker, config.rerank_candidates)

    results: list[RetrievedChunk] = []
    for position, (chunk_id, score, rerank_score) in enumerate(ordered[: config.top_k], start=1):
        results.append(
            RetrievedChunk(
                chunk=by_id[chunk_id],
                score=score,
                rank=position,
                dense_rank=dense_ranks.get(chunk_id),
                fts_rank=fts_ranks.get(chunk_id),
                rerank_score=rerank_score,
            )
        )
    return results


def _order(
    config: RetrievalConfig,
    dense_hits: Sequence[ScoredChunk],
    fts_hits: Sequence[ScoredChunk],
) -> list[tuple[str, float, float | None]]:
    """Produce (chunk_id, score, rerank_score) in final order before reranking."""
    if config.retriever == "dense":
        return [(hit.chunk.chunk_id, hit.score, None) for hit in dense_hits]
    if config.retriever == "fts":
        return [(hit.chunk.chunk_id, hit.score, None) for hit in fts_hits]

    fused = fuse_and_order(
        [
            [hit.chunk.chunk_id for hit in dense_hits],
            [hit.chunk.chunk_id for hit in fts_hits],
        ],
        k=config.rrf_k,
    )
    return [(chunk_id, score, None) for chunk_id, score in fused]


def _rerank(
    query: str,
    ordered: Sequence[tuple[str, float, float | None]],
    by_id: dict[str, Chunk],
    reranker: Reranker,
    candidates: int,
) -> list[tuple[str, float, float | None]]:
    """Rescore the top candidates with the cross-encoder.

    Only the head of the list is rescored; the tail keeps its first-stage order
    and sits below every reranked item, because a cross-encoder score and an
    RRF score are not comparable.
    """
    head = list(ordered[:candidates])
    tail = list(ordered[candidates:])
    if not head:
        return tail

    scores = reranker.score(query, [by_id[chunk_id].text for chunk_id, _, _ in head])
    rescored = [
        (chunk_id, float(score), float(score))
        for (chunk_id, _, _), score in zip(head, scores, strict=True)
    ]
    rescored.sort(key=lambda item: (-item[1], item[0]))
    return rescored + tail
