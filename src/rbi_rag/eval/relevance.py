"""Deciding which retrieved chunks count as relevant.

The ablation compares six chunking configurations. A chunk id encodes its
configuration, and no id survives a re-chunk -- measured: zero overlap between
any two configurations. So relevance cannot be a set membership test against
gold chunk ids; it has to be computed from where the chunk sits in the source
document.

A chunk covers a gold span when it comes from the same document, its page range
overlaps the span's, and -- when the span carries a quote -- that quote appears
in the chunk's text. The quote is what makes a span precise: page overlap alone
is generous on a long page, and a 1024-token chunk would otherwise "cover" a
span it merely sits near.
"""

from __future__ import annotations

from collections.abc import Sequence

from rbi_rag.generate.validate import normalise
from rbi_rag.schemas import Chunk, GoldSpan, RetrievedChunk


def chunk_covers_span(chunk: Chunk, span: GoldSpan) -> bool:
    """True when this chunk contains the evidence the span points at."""
    if chunk.doc_id != span.doc_id:
        return False
    if not span.covers_pages(chunk.page_start, chunk.page_end):
        return False
    if span.quote is None:
        return True
    return normalise(span.quote) in normalise(chunk.text)


def spans_covered(chunk: Chunk, spans: Sequence[GoldSpan]) -> set[int]:
    """Indices of the gold spans this chunk covers."""
    return {index for index, span in enumerate(spans) if chunk_covers_span(chunk, span)}


def coverage_sequence(
    retrieved: Sequence[RetrievedChunk],
    spans: Sequence[GoldSpan],
) -> list[set[int]]:
    """Per rank, the set of gold-span indices that chunk covers."""
    return [spans_covered(item.chunk, spans) for item in retrieved]


def novel_coverage_sequence(
    retrieved: Sequence[RetrievedChunk],
    spans: Sequence[GoldSpan],
) -> list[set[int]]:
    """Per rank, the gold spans this chunk is the *first* to cover.

    Ranking metrics should reward surfacing a new piece of evidence, not
    repeating one already found higher up. Without this, a configuration that
    splits one span across many small chunks would score better simply for
    returning the same evidence several times.
    """
    seen: set[int] = set()
    novel: list[set[int]] = []
    for covered in coverage_sequence(retrieved, spans):
        fresh = covered - seen
        seen |= fresh
        novel.append(fresh)
    return novel


def resolve_gold_chunk_ids(
    chunks: Sequence[Chunk],
    spans: Sequence[GoldSpan],
) -> list[str]:
    """The chunk ids a gold span resolves to in one particular chunk set.

    Derived output, never a label: it lets a result be inspected ("which chunks
    was this arm supposed to find?") without the eval set itself depending on a
    chunking configuration.
    """
    return sorted(chunk.chunk_id for chunk in chunks if spans_covered(chunk, spans))
