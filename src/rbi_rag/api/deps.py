"""Wiring for the API.

Everything expensive -- the ONNX sessions, the Anthropic client -- is built
once and held on the app, never per request. Every piece is overridable so the
tests can run the real request path against fakes, with no database, no model
downloads and no API key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rbi_rag.config import Settings
from rbi_rag.generate.client import LLMClient
from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import ChunkStore, PostgresChunkStore, connect
from rbi_rag.retrieve.rerank import Reranker

StoreProvider = Callable[[], Any]


@dataclass
class AppState:
    """What a request needs, resolved once at startup."""

    settings: Settings
    store_provider: StoreProvider
    llm: LLMClient | None = None
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    chunk_set_resolver: Callable[[], int] | None = None

    def resolve_chunk_set(self) -> int:
        if self.chunk_set_resolver is None:
            raise RuntimeError("no chunk set resolver configured")
        return self.chunk_set_resolver()


def postgres_store_provider(settings: Settings) -> StoreProvider:
    """Open a connection per request.

    Deliberately not pooled: psycopg_pool is a dependency this project has not
    agreed to, and a connection costs single-digit milliseconds against a
    generation call measured in seconds. Phase 6 should revisit it if the API
    is ever put under real concurrency.
    """

    @contextmanager
    def provider() -> Iterator[ChunkStore]:
        with connect(settings.postgres_dsn) as connection:
            yield PostgresChunkStore(connection)

    return provider


def default_chunk_set_resolver(settings: Settings) -> Callable[[], int]:
    """Resolve the chunk set to search, preferring an explicit configuration."""

    def resolve() -> int:
        from rbi_rag.index.build import list_chunk_sets

        with connect(settings.postgres_dsn) as connection:
            sets = list_chunk_sets(connection)
        if not sets:
            raise RuntimeError("no chunk sets in the database; run `uv run poe index build`")
        if len(sets) > 1:
            available = ", ".join(str(row["chunk_set_id"]) for row in sets)
            raise RuntimeError(
                f"several chunk sets exist ({available}); pass chunk_set_id on the request"
            )
        return int(sets[0]["chunk_set_id"])

    return resolve
