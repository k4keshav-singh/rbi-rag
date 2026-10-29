"""The shared models carry citation metadata, so their invariants matter."""

import pytest
from pydantic import ValidationError

from rbi_rag.schemas import Chunk, Citation


def make_chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "Banks shall maintain a minimum CRAR of 9 per cent.",
        "page_start": 3,
        "page_end": 3,
        "circular_number": "RBI/2023-24/01",
        "url": "https://example.test/md.pdf",
        "strategy": "fixed_tokens",
        "target_tokens": 512,
    }
    return Chunk(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_chunk_round_trips() -> None:
    chunk = make_chunk()
    assert Chunk.model_validate(chunk.model_dump()) == chunk


def test_chunk_is_frozen() -> None:
    """Chunks are compared and hashed during eval; mutation would corrupt that."""
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.text = "something else"  # type: ignore[misc]


def test_pages_are_one_indexed() -> None:
    with pytest.raises(ValidationError):
        make_chunk(page_start=0)


def test_unknown_chunk_strategy_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_chunk(strategy="semantic")


def test_citation_requires_a_source_url() -> None:
    with pytest.raises(ValidationError):
        Citation(chunk_id="c1", page_start=1, page_end=1)  # type: ignore[call-arg]
