"""Command line entry point for retrieval.

    uv run poe query "minimum capital adequacy ratio"
    uv run poe query "..." --retriever dense --top-k 5 --rerank
    uv run poe query "..." --exact            # bypass HNSW, exact kNN

Prints the per-retriever ranks next to the final score, so it is visible *why*
a chunk ranked where it did rather than only that it did.
"""

from __future__ import annotations

import argparse
import textwrap
import time
from collections.abc import Sequence
from typing import cast

from rbi_rag.config import get_settings
from rbi_rag.index.factory import build_embedder, build_reranker
from rbi_rag.index.store import PostgresChunkStore, connect
from rbi_rag.retrieve.engine import retrieve
from rbi_rag.schemas import RetrievalConfig, RetrievedChunk, RetrieverName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbi-rag query", description=__doc__)
    parser.add_argument("query", help="the question to retrieve for")
    parser.add_argument("--retriever", choices=("dense", "fts", "hybrid"))
    parser.add_argument("--chunk-set", type=int, help="chunk set id (default: the only one)")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-k", type=int, help="depth per retriever before fusion")
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--exact", action="store_true", help="bypass the HNSW index and do exact kNN"
    )
    parser.add_argument("--chars", type=int, default=240, help="characters of text to show")
    return parser


def resolve_chunk_set(connection: object, requested: int | None) -> int:
    """Pick the chunk set to search, or explain why we cannot."""
    from rbi_rag.index.build import list_chunk_sets

    sets = list_chunk_sets(connection)
    if not sets:
        raise SystemExit("no chunk sets in the database; run `uv run poe index build` first")
    if requested is not None:
        if not any(row["chunk_set_id"] == requested for row in sets):
            available = ", ".join(str(row["chunk_set_id"]) for row in sets)
            raise SystemExit(f"chunk set {requested} does not exist; available: {available}")
        return requested
    if len(sets) > 1:
        available = ", ".join(
            f"{row['chunk_set_id']} ({row['strategy']}/{row['target_tokens']})" for row in sets
        )
        raise SystemExit(f"several chunk sets exist; choose one with --chunk-set: {available}")
    return int(sets[0]["chunk_set_id"])


def format_result(result: RetrievedChunk, chars: int) -> str:
    chunk = result.chunk
    pages = (
        f"p{chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"pp{chunk.page_start}-{chunk.page_end}"
    )
    provenance = " ".join(
        part
        for part in (
            f"dense#{result.dense_rank}" if result.dense_rank else "",
            f"fts#{result.fts_rank}" if result.fts_rank else "",
            f"rerank={result.rerank_score:.3f}" if result.rerank_score is not None else "",
        )
        if part
    )
    header = (
        f"{result.rank:>2}. score={result.score:.5f}  {provenance or 'no first-stage rank'}\n"
        f"    {chunk.circular_number or chunk.doc_id} | {pages}"
    )
    if chunk.section_title:
        header += f" | {chunk.section_title[:60]}"
    body = textwrap.fill(
        " ".join(chunk.text.split())[:chars],
        width=92,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    return f"{header}\n{body}\n    {chunk.url}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with connect(settings.postgres_dsn) as connection:
        chunk_set_id = resolve_chunk_set(connection, args.chunk_set)
        config = RetrievalConfig(
            retriever=cast("RetrieverName", args.retriever or settings.retriever),
            chunk_set_id=chunk_set_id,
            top_k=args.top_k or settings.top_k,
            candidate_k=args.candidate_k or settings.candidate_k,
            rrf_k=args.rrf_k or settings.rrf_k,
            rerank=args.rerank,
            rerank_candidates=settings.rerank_candidates,
            use_query_instruction=settings.use_query_instruction,
            exact=args.exact or settings.exact_search,
        )

        embedder = build_embedder(settings, for_queries=True) if config.retriever != "fts" else None
        reranker = build_reranker(settings) if config.rerank else None

        started = time.perf_counter()
        results = retrieve(
            args.query,
            store=PostgresChunkStore(connection),
            config=config,
            embedder=embedder,
            reranker=reranker,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

    search = "exact" if config.exact else "hnsw"
    print(
        f"{config.retriever} | chunk set {chunk_set_id} | {search} | "
        f"{len(results)} result(s) in {elapsed_ms:.0f} ms\n"
    )
    if not results:
        print("nothing matched.")
        return 0
    for result in results:
        print(format_result(result, args.chars))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
