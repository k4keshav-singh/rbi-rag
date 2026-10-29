"""Data models shared across ingest, index, retrieve, generate and eval.

These live in one place so that no two packages need to import each other to
agree on a type. Fields are added as the phase that needs them lands; nothing
here is speculative.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkStrategy = Literal["fixed_tokens", "section_aware"]
QuestionType = Literal["factual", "multi_hop", "unanswerable"]


class Document(BaseModel):
    """One RBI Master Direction PDF as downloaded, before parsing."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Stable id derived from the circular number.")
    title: str
    circular_number: str | None = None
    published_on: date | None = None
    department: str | None = None
    url: str
    sha256: str = Field(description="Hash of the raw PDF bytes; drives idempotent re-ingest.")


class Chunk(BaseModel):
    """A retrievable span of text with everything needed to cite it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    circular_number: str | None = None
    url: str
    strategy: ChunkStrategy
    target_tokens: int = Field(ge=1)


class RetrievedChunk(BaseModel):
    """A chunk together with the scores that put it in the result set."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Score of the retriever that produced the final ranking.")
    rank: int = Field(ge=1)
    dense_rank: int | None = None
    fts_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """A pointer the reader can verify against the source PDF."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    circular_number: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    url: str
    quote: str | None = None
