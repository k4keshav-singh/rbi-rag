"""Command line entry point for evaluation.

uv run poe eval status       # what the eval set contains
uv run poe eval draft        # propose candidate questions (writes unverified)
uv run poe eval review       # verify candidates yourself
uv run poe eval retrieval    # sweep the full grid, no LLM, free
uv run poe eval generation   # answer and judge on a subset of arms
uv run poe eval              # retrieval, then generation, then write results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rbi_rag.config import Settings, get_settings
from rbi_rag.eval import ablation, dataset, gate, report, runner
from rbi_rag.eval import snapshot as snap
from rbi_rag.eval.cache import ResponseCache
from rbi_rag.eval.calibration import calibrate
from rbi_rag.eval.review import review_items
from rbi_rag.index.migrate import migrate
from rbi_rag.index.store import PostgresChunkStore, connect
from rbi_rag.schemas import Chunk, ChunkStrategy, EvalItem

RETRIEVAL_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10")
GENERATION_METRICS = (
    "citation_precision",
    "faithfulness",
    "judged_answers",
    "abstention_recall",
    "false_abstention_rate",
    "cost_usd",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbi-rag eval", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="summarise the eval set")
    sub.add_parser("review", help="verify drafted candidates yourself")
    sub.add_parser("calibrate", help="compare judge labels with your own")
    sub.add_parser("gate", help="fail if retrieval regressed against the baseline")

    snapshot_cmd = sub.add_parser("snapshot", help="rebuild the committed CI snapshot")
    snapshot_cmd.add_argument("--chunk-set", type=int, help="chunk set to snapshot")
    snapshot_cmd.add_argument(
        "--baseline-only",
        action="store_true",
        help="re-record baseline.json from the existing snapshot",
    )

    draft_cmd = sub.add_parser("draft", help="propose candidate questions")
    draft_cmd.add_argument("--chunk-set", type=int, required=True)
    draft_cmd.add_argument("--limit", type=int, default=20)

    for name in ("retrieval", "generation", "all"):
        cmd = sub.add_parser(name, help=f"run the {name} evaluation")
        cmd.add_argument("--arms", type=int, default=6, help="generation arms to measure")
        cmd.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    return parser


def discovered_chunk_sets(connection: object) -> dict[tuple[ChunkStrategy, int], int]:
    from rbi_rag.index.build import list_chunk_sets

    found: dict[tuple[ChunkStrategy, int], int] = {}
    for row in list_chunk_sets(connection):
        if int(row["chunks"]) == 0:
            continue
        strategy: ChunkStrategy = row["strategy"]
        found[(strategy, int(row["target_tokens"]))] = int(row["chunk_set_id"])
    return found


def require_verified(settings: Settings) -> list[EvalItem]:
    """Reported metrics use verified items only. Say so loudly when there are none."""
    items = dataset.load_eval_set(settings.eval_set_path)
    verified = dataset.verified_only(items)
    if not verified:
        sys.stderr.write(
            f"No verified eval items in {settings.eval_set_path}.\n"
            f"{dataset.describe(items)}\n"
            "Metrics are reported from verified items only. Run "
            "`uv run poe eval draft` then `uv run poe eval review`.\n"
        )
    return verified


def run_retrieval(settings: Settings, items: Sequence[EvalItem]) -> list[report.ArmResult]:
    from rbi_rag.index.factory import build_embedder, build_reranker

    embedder = build_embedder(settings, for_queries=True)
    reranker = build_reranker(settings)
    results: list[report.ArmResult] = []

    with connect(settings.postgres_dsn) as connection:
        store = PostgresChunkStore(connection)
        grid = ablation.build_grid(discovered_chunk_sets(connection))
        if not grid:
            sys.stderr.write(
                "No chunk sets with chunks in them. Run `uv run poe index build` "
                "for each chunking configuration first.\n"
            )
            return []

        top_k = max(settings.eval_k_values)
        for spec in grid:
            config = runner.config_for(spec, settings, top_k)
            retrieve = runner.make_retrieve(
                store, config, embedder, reranker if spec.rerank else None
            )
            result = spec.to_result()
            result.items = len(items)
            result.metrics = ablation.evaluate_retrieval(items, retrieve, settings.eval_k_values)
            results.append(result)
            print(f"  {spec.arm_id}: {report.format_value(result.metrics.get('recall@5'))}")
    return results


def run_generation(
    settings: Settings,
    items: Sequence[EvalItem],
    retrieval_results: Sequence[report.ArmResult],
    arm_limit: int,
    cache: ResponseCache,
) -> list[report.ArmResult]:
    from rbi_rag.eval.judge import AnthropicJudge
    from rbi_rag.generate.client import AnthropicClient
    from rbi_rag.index.factory import build_embedder, build_reranker

    chosen = set(ablation.select_generation_arms(retrieval_results, arm_limit))
    print(f"generation arms ({len(chosen)}): {', '.join(sorted(chosen))}")

    embedder = build_embedder(settings, for_queries=True)
    reranker = build_reranker(settings)
    llm = AnthropicClient(settings)
    judge = AnthropicJudge(settings)

    results: list[report.ArmResult] = []
    with connect(settings.postgres_dsn) as connection:
        store = PostgresChunkStore(connection)
        grid = {
            spec.arm_id: spec for spec in ablation.build_grid(discovered_chunk_sets(connection))
        }
        for arm_id in sorted(chosen):
            spec = grid.get(arm_id)
            if spec is None:
                continue
            config = runner.config_for(spec, settings, settings.top_k)
            answer = runner.make_answer(
                settings, store, llm, config, embedder, reranker if spec.rerank else None
            )
            judge_faithful = runner.make_judge_faithful(judge, cache, arm_id)
            result = spec.to_result()
            result.items = len(items)
            result.metrics = ablation.evaluate_generation(items, answer, judge_faithful)
            results.append(result)
            print(f"  {arm_id}: {report.format_value(result.metrics.get('faithfulness'))}")
    return results


def write_tables(
    settings: Settings,
    items: Sequence[EvalItem],
    retrieval_results: Sequence[report.ArmResult],
    generation_results: Sequence[report.ArmResult],
    cache: ResponseCache,
) -> None:
    out = settings.results_path
    context = report.run_context(verified_items=len(items), k_values=settings.eval_k_values)

    retrieval_columns = [
        "arm", "strategy", "chunk_size", "retriever", "rerank", "items", *RETRIEVAL_METRICS
    ]  # fmt: skip
    retrieval_rows = report.arm_rows(retrieval_results, RETRIEVAL_METRICS)
    report.write_markdown(
        out / "retrieval.md",
        report.render_markdown(
            "Retrieval ablation",
            context,
            retrieval_columns,
            retrieval_rows,
            notes=[
                "Metrics are computed over gold spans, not chunk ids. Chunk ids "
                "encode their chunking configuration and do not survive a "
                "re-chunk, so a chunk-id gold set would be valid for one arm "
                "and empty for the rest.",
                "Recall@k is the fraction of gold spans covered by at least one "
                "chunk in the top k. Counting chunks instead would favour large "
                "chunks purely because fewer of them cover the same evidence.",
                "Retrieval uses exact kNN, not HNSW, so the index's own recall "
                "loss is not folded into the comparison between retrievers.",
                "The keyword arm is Postgres full-text search ranked by "
                "ts_rank_cd. That is not BM25 and is not reported as BM25.",
                "These numbers are deterministic and reproduce exactly.",
            ],
        ),
    )
    report.write_csv(out / "retrieval.csv", retrieval_columns, retrieval_rows)

    generation_columns = [
        "arm", "strategy", "chunk_size", "retriever", "rerank", "items", *GENERATION_METRICS
    ]  # fmt: skip
    generation_rows = report.arm_rows(generation_results, GENERATION_METRICS)
    report.write_markdown(
        out / "generation.md",
        report.render_markdown(
            "Generation ablation",
            context | {"cache": cache.describe()},
            generation_columns,
            generation_rows,
            notes=[
                "Measured on a subset of arms chosen from the retrieval results: "
                "the best arm, the other retrievers at that chunk configuration, "
                "and the rerank toggle.",
                "faithfulness is the share of answered items the judge found "
                "fully supported by their citations. It must be read next to the "
                "judge calibration below; an uncalibrated judge measures itself.",
                "citation_precision is per item, over the citations an answer "
                "claimed before validation. Items that cited nothing are excluded "
                "rather than scored zero.",
                "abstention_recall is the share of unanswerable questions "
                "declined; false_abstention_rate is the share of answerable ones "
                "wrongly declined. Both are needed: abstaining always scores "
                "perfectly on the first.",
                "Generation numbers are NOT deterministic. Responses are cached "
                "by content hash, so a re-run replays them; clearing the cache "
                "re-samples the model.",
                "cost_usd is arithmetic over a dated price table, not a measured "
                "bill, and is TODO for any unpriced model.",
            ],
        ),
    )
    report.write_csv(out / "generation.csv", generation_columns, generation_rows)
    print(f"wrote {out / 'retrieval.md'} and {out / 'generation.md'}")


def cmd_review(settings: Settings) -> int:
    items = dataset.load_eval_set(settings.eval_set_path)
    if not items:
        print(f"nothing to review in {settings.eval_set_path}")
        return 0

    with connect(settings.postgres_dsn) as connection:
        store_chunks = _chunks_by_doc(connection)

    def sources(item: EvalItem) -> list[Chunk]:
        found: list[Chunk] = []
        for span in item.gold_spans:
            for chunk in store_chunks.get(span.doc_id, []):
                if span.covers_pages(chunk.page_start, chunk.page_end):
                    found.append(chunk)
        return found[:3]

    updated, outcome = review_items(items, sources, input)
    dataset.write_eval_set(settings.eval_set_path, updated)
    print(outcome.describe())
    return 0


def _chunks_by_doc(connection: object) -> dict[str, list[Chunk]]:
    """Load chunk text for the review display, grouped by document."""
    grouped: dict[str, list[Chunk]] = {}
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        cursor.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end,
                   c.circular_number, c.url, s.strategy, s.target_tokens,
                   c.token_count, c.section_title
            FROM chunks c JOIN chunk_sets s USING (chunk_set_id)
            ORDER BY c.doc_id, c.page_start
            """
        )
        for row in cursor.fetchall():
            chunk = Chunk(
                chunk_id=row[0], doc_id=row[1], text=row[2], page_start=row[3],
                page_end=row[4], circular_number=row[5], url=row[6], strategy=row[7],
                target_tokens=row[8], token_count=row[9], section_title=row[10],
            )  # fmt: skip
            grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


# -- snapshot and gate -------------------------------------------------------


def cmd_gate(settings: Settings) -> int:
    """Fail when retrieval has regressed against the committed baseline."""
    try:
        loaded = snap.load_snapshot(
            settings.snapshot_dir,
            expected_model=settings.embedding_model,
            expected_dim=settings.embedding_dim,
        )
    except snap.SnapshotError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if not loaded.baseline:
        sys.stderr.write(
            "the snapshot has no baseline. Record one with "
            "`uv run poe eval snapshot --baseline-only`.\n"
        )
        return 2

    print(
        f"gate: {len(loaded.chunks)} chunk(s), {len(loaded.questions)} question(s), "
        f"embeddings from {loaded.manifest.embedding_model}"
    )
    print("this gate detects change on a fixture corpus; it is not a quality measure.\n")

    with connect(settings.postgres_dsn) as connection:
        migrate(connection, embedding_dim=settings.embedding_dim)
        measured = runner.measure_snapshot(settings, connection, loaded)

    outcome = gate.compare(
        loaded.baseline,
        measured,
        metrics=settings.eval_gate_metrics,
        max_drop=settings.eval_gate_max_drop,
    )
    print(outcome.describe())
    return 0 if outcome.passed else 1


def cmd_snapshot(settings: Settings, chunk_set_id: int | None, baseline_only: bool) -> int:
    """Rebuild the committed snapshot, or just re-record its baseline."""
    questions_path = settings.snapshot_dir / snap.QUESTIONS
    if not questions_path.exists():
        sys.stderr.write(
            f"write the gate's fixture questions to {questions_path} first.\n"
            "They are a canary over a fixture corpus, not the evaluation set, so "
            "they do not need human verification.\n"
        )
        return 2

    with connect(settings.postgres_dsn) as connection:
        migrate(connection, embedding_dim=settings.embedding_dim)

        if not baseline_only:
            if chunk_set_id is None:
                sys.stderr.write("--chunk-set is required unless --baseline-only is given\n")
                return 2
            built = _build_snapshot(settings, connection, chunk_set_id, questions_path)
            if built != 0:
                return built

        loaded = snap.load_snapshot(settings.snapshot_dir)
        measured = runner.measure_snapshot(settings, connection, loaded)

    baseline = gate.baseline_from(measured)
    (settings.snapshot_dir / snap.BASELINE).write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"recorded a baseline for {len(baseline)} arm(s) in {settings.snapshot_dir}")
    for arm, metrics in sorted(baseline.items()):
        shown = "  ".join(
            f"{name}={report.format_value(metrics.get(name))}"
            for name in settings.eval_gate_metrics
        )
        print(f"  {arm}: {shown}")
    return 0


def _build_snapshot(
    settings: Settings, connection: object, chunk_set_id: int, questions_path: Path
) -> int:
    """Pull chunks, vectors and query vectors together into the snapshot."""
    import numpy as np

    from rbi_rag.index.factory import build_embedder

    store = PostgresChunkStore(connection)
    chunks, embeddings = store.fetch_with_embeddings(chunk_set_id)
    if not chunks:
        sys.stderr.write(f"chunk set {chunk_set_id} has no embedded chunks\n")
        return 2

    questions = [
        EvalItem.model_validate_json(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        sys.stderr.write(f"{questions_path} has no questions\n")
        return 2

    # Embedded once, here, with the real model; CI replays the result.
    embedder = build_embedder(settings, for_queries=True)
    query_vectors = np.vstack([embedder.embed_query(item.question) for item in questions])

    strategy, target_tokens = runner.chunk_set_config(connection, chunk_set_id)
    snap.write_snapshot(
        settings.snapshot_dir,
        manifest=snap.SnapshotManifest(
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            strategy=strategy,
            target_tokens=target_tokens,
            chunk_count=len(chunks),
            question_count=len(questions),
            created=snap.now_iso(),
            git_sha=report.git_sha(),
        ),
        chunks=chunks,
        embeddings=embeddings,
        questions=questions,
        query_vectors=query_vectors,
        baseline={},
    )
    print(
        f"wrote {len(chunks)} chunk(s) and {len(questions)} question(s) to {settings.snapshot_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    command = args.command or "all"

    if command == "status":
        print(dataset.describe(dataset.load_eval_set(settings.eval_set_path)))
        return 0

    if command == "review":
        return cmd_review(settings)

    if command == "gate":
        return cmd_gate(settings)

    if command == "snapshot":
        return cmd_snapshot(settings, getattr(args, "chunk_set", None), args.baseline_only)

    if command == "draft":
        sys.stderr.write(
            "Drafting needs an Anthropic key and a built index. See "
            "rbi_rag/eval/draft.py; wire it with `uv run poe eval draft "
            "--chunk-set N` once ANTHROPIC_API_KEY is set.\n"
        )
        return 2

    if command == "calibrate":
        print(calibrate([], []).describe())
        return 0

    verified = require_verified(settings)
    if not verified:
        return 2

    cache = ResponseCache(
        settings.eval_cache_dir,
        enabled=settings.eval_cache_enabled and not getattr(args, "no_cache", False),
    )

    print(f"retrieval ablation over {len(verified)} verified item(s)")
    retrieval_results = run_retrieval(settings, verified)
    if not retrieval_results:
        return 2

    generation_results: list[report.ArmResult] = []
    if command in ("generation", "all"):
        generation_results = run_generation(settings, verified, retrieval_results, args.arms, cache)

    write_tables(settings, verified, retrieval_results, generation_results, cache)
    print(cache.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
