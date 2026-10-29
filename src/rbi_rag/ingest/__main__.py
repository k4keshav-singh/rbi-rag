"""Command line entry point for ingestion.

uv run poe ingest harvest     # fetch the listing, update the manifest
uv run poe ingest urls        # write the list of PDFs still to download
uv run poe ingest status      # hash whatever is in data/raw, report gaps
uv run poe ingest process     # parse, clean and chunk what is present
uv run poe ingest all         # the three above, in order
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import cast

from rbi_rag.config import get_settings
from rbi_rag.ingest import pipeline
from rbi_rag.ingest.tokenize import (
    HuggingFaceTokenCounter,
    TokenCounter,
    WhitespaceTokenCounter,
)
from rbi_rag.schemas import ChunkStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbi-rag ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("harvest", help="fetch the RBI listing page and update the manifest")
    sub.add_parser("urls", help="write the URLs of PDFs that are not yet downloaded")
    sub.add_parser("status", help="hash the raw store and report what is missing")

    for name in ("process", "all"):
        step = sub.add_parser(name, help=f"{name} the corpus")
        step.add_argument(
            "--strategy",
            choices=("fixed_tokens", "section_aware"),
            help="chunking strategy (default: CHUNK_STRATEGY)",
        )
        step.add_argument("--size", type=int, help="target tokens per chunk (default: CHUNK_SIZE)")
        step.add_argument(
            "--token-counter",
            choices=("model", "whitespace"),
            default="model",
            help=(
                "'model' uses the embedding model's tokenizer and is the only one "
                "whose counts may be reported; 'whitespace' is an offline approximation"
            ),
        )
        step.add_argument("--force", action="store_true", help="re-chunk even if unchanged")
        step.add_argument("--limit", type=int, help="process at most N documents")

    return parser


def _counter(choice: str, model_name: str) -> TokenCounter:
    if choice == "whitespace":
        sys.stderr.write(
            "warning: counting tokens by whitespace. These counts are an approximation "
            "and must not appear in reported results.\n"
        )
        return WhitespaceTokenCounter()
    return HuggingFaceTokenCounter(model_name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "harvest":
        print(pipeline.run_harvest(settings).describe())
        return 0

    if args.command == "urls":
        path, count = pipeline.write_pdf_url_list(settings)
        print(f"{count} PDF(s) still to download; URLs written to {path}")
        return 0

    if args.command == "status":
        print(pipeline.refresh_acquisition(settings).describe())
        return 0

    if args.command == "all":
        print(pipeline.run_harvest(settings).describe())
        print(pipeline.refresh_acquisition(settings).describe())

    report = pipeline.run_process(
        settings,
        _counter(args.token_counter, settings.embedding_model),
        strategy=cast("ChunkStrategy | None", args.strategy),
        target_tokens=args.size,
        force=args.force,
        limit=args.limit,
    )
    print(report.describe())
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
