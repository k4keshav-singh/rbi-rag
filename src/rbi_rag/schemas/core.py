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
RetrieverName = Literal["dense", "fts", "hybrid"]


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
    dense_score: float | None = Field(
        default=None,
        description=(
            "Raw cosine similarity, kept even under hybrid. `score` is an RRF value "
            "there and is not comparable across retrievers; this is."
        ),
    )


class Citation(BaseModel):
    """A pointer the reader can verify against the source PDF."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    circular_number: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    url: str
    quote: str | None = None


class RetrievalConfig(BaseModel):
    """Everything that decides how a query is answered.

    The Phase 4 ablation varies these and nothing else, so a run is fully
    described by one of these objects plus the chunk set it points at.
    """

    model_config = ConfigDict(frozen=True)

    retriever: RetrieverName = "hybrid"
    chunk_set_id: int = Field(ge=1)
    top_k: int = Field(default=10, ge=1)
    candidate_k: int = Field(
        default=50, ge=1, description="Depth fetched from each retriever before fusion."
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant.")
    rerank: bool = False
    rerank_candidates: int = Field(default=50, ge=1)
    use_query_instruction: bool = False
    exact: bool = Field(
        default=False,
        description="Bypass the HNSW index for exact kNN. Evaluation sets this.",
    )


class CitationClaim(BaseModel):
    """A citation as the model asserts it, before anything has been checked.

    Deliberately distinct from Citation: this is an unverified claim. It only
    becomes a Citation after validate.py confirms the chunk was retrieved and
    the quote actually appears in it.
    """

    chunk_id: str = Field(description="Id of a chunk from the supplied context.")
    quote: str = Field(description="Verbatim span from that chunk supporting the claim.")


class AnswerPayload(BaseModel):
    """The structured output contract with the model."""

    answer: str = Field(description="The answer, or an empty string when abstaining.")
    citations: list[CitationClaim] = Field(default_factory=list)
    sufficient_evidence: bool = Field(
        description="False when the supplied circulars do not answer the question."
    )


class TokenUsage(BaseModel):
    """Tokens consumed by one generation call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class QueryRequest(BaseModel):
    """A question, plus any retrieval overrides for this one request."""

    question: str = Field(min_length=1, max_length=2000)
    retriever: RetrieverName | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    chunk_set_id: int | None = Field(default=None, ge=1)


class QueryResponse(BaseModel):
    """What POST /query returns."""

    answer: str
    abstained: bool = Field(description="True when the system declined to answer.")
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    usage: TokenUsage = TokenUsage()
    cost_usd: float | None = Field(
        default=None,
        description="None when the model has no entry in the price table, never a guess.",
    )
    model: str
    retriever: RetrieverName


class GoldSpan(BaseModel):
    """Where the evidence for an answer actually lives, in the source document.

    Deliberately NOT a chunk id. Chunk ids encode the chunking configuration
    (`{doc_id}-{strategy}-{size}-{index}-{hash}`), so a gold set keyed on them
    is valid for exactly one arm of the ablation and empty for every other --
    measured: zero id overlap between any two chunk configurations. Labelling
    by document, page range and quote is chunk-set independent, so one label
    serves every arm. See DECISIONS.md entry 0026.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    quote: str | None = Field(
        default=None,
        description="Verbatim text that must appear in a chunk for it to count.",
    )

    def covers_pages(self, page_start: int, page_end: int) -> bool:
        """True when the given page range overlaps this span at all."""
        return page_start <= self.page_end and page_end >= self.page_start


class EvalItem(BaseModel):
    """One evaluation question.

    `verified_by_human` is false for anything a script produced and may only be
    set by the repository owner through the review CLI. Every reported metric
    filters on it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str = Field(min_length=1)
    question_type: QuestionType
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    gold_answer_notes: str = ""
    verified_by_human: bool = False
    drafted_by: str | None = Field(
        default=None, description="Model that proposed this item, or None if hand-written."
    )
    drafted_at: datetime | None = None
    source_doc_id: str | None = None

    @property
    def is_answerable(self) -> bool:
        return self.question_type != "unanswerable"
