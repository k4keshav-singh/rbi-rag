"""Run the API.

uv run poe serve
uv run poe serve --reload --port 8080
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rbi_rag.api.logging import configure_logging
from rbi_rag.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbi-rag serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="reload on code changes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    import uvicorn

    uvicorn.run(
        "rbi_rag.api.factory:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,  # our JSON formatter owns the root logger
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
