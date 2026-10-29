"""Data models shared across ingest, index, retrieve, generate and eval.

These live in one place so that no two packages need to import each other to
agree on a type. Fields are added as the phase that needs them lands; nothing
here is speculative.
"""

from __future__ import annotations

from datetime import date, datetime
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


class ManifestEntry(BaseModel):
    """One row of the ingest manifest.

    Written by the metadata harvester from rbi.org.in, then enriched in place
    once the corresponding PDF is present in the raw store. Acquisition and
    processing state are separate so a run can resume from either.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="RBI's own detail-page id; stable across re-harvests.")
    title: str
    detail_url: str
    pdf_url: str
    category: str | None = Field(
        default=None, description="Listing section heading, e.g. 'Commercial Banks'."
    )
    circular_number: str | None = None
    published_on: date | None = None
    department: str | None = None

    # Populated once the PDF is in the raw store.
    pdf_filename: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    acquired_at: datetime | None = None

    @property
    def is_acquired(self) -> bool:
        """True once the PDF bytes exist locally and have been hashed."""
        return self.sha256 is not None


class PageText(BaseModel):
    """Text of a single PDF page, one-indexed to match what a reader sees."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str


class Section(BaseModel):
    """A titled span of a document, used by the section_aware chunker."""

    model_config = ConfigDict(frozen=True)

    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str


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
    token_count: int | None = Field(
        default=None, description="Actual tokens under the embedding model's tokenizer."
    )
    section_title: str | None = None


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
