"""Command line entry point for indexing.

uv run poe index migrate    # apply outstanding SQL migrations
uv run poe index build      # embed a chunk set and index it
uv run poe index sets       # list the chunk sets in the database
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast

from rbi_rag.config import get_settings
from rbi_rag.index import build as builder
from rbi_rag.index.factory import build_embedder
from rbi_rag.index.migrate import migrate
from rbi_rag.index.store import connect
from rbi_rag.schemas import ChunkStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbi-rag index", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply outstanding migrations")
    sub.add_parser("sets", help="list chunk sets already in the database")

    build_cmd = sub.add_parser("build", help="embed and index a chunk set")
    build_cmd.add_argument(
        "--strategy",
        choices=("fixed_tokens", "section_aware"),
        help="chunking strategy (default: CHUNK_STRATEGY)",
    )
    build_cmd.add_argument("--size", type=int, help="target tokens per chunk (default: CHUNK_SIZE)")
    build_cmd.add_argument(
        "--no-hnsw",
        action="store_true",
        help="skip the HNSW index; exact search still works and stays exact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    with connect(settings.postgres_dsn) as connection:
        applied = migrate(connection, embedding_dim=settings.embedding_dim)
        if applied:
            print(f"applied migrations: {', '.join(applied)}")
        elif args.command == "migrate":
            print("schema is up to date")

        if args.command == "migrate":
            return 0

        if args.command == "sets":
            rows = builder.list_chunk_sets(connection)
            if not rows:
                print("no chunk sets yet; run `uv run poe index build`")
                return 0
            print(f"{'id':>3}  {'strategy':<14} {'size':>5} {'chunks':>7}  model")
            for row in rows:
                print(
                    f"{row['chunk_set_id']:>3}  {row['strategy']:<14} "
                    f"{row['target_tokens']:>5} {row['chunks']:>7}  {row['embedding_model']}"
                )
            return 0

        embedder = build_embedder(settings)
        report = builder.build_index(
            settings,
            connection,
            embedder,
            strategy=cast("ChunkStrategy | None", args.strategy),
            target_tokens=args.size,
            create_index=not args.no_hnsw,
        )
        print(report.describe())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
