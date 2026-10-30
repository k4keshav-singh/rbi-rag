"""Citation validation.

Two checks, both of which must pass for a citation to survive:

1. **The chunk was retrieved.** A chunk_id the model was never shown means it
   invented an identifier.
2. **The quote is really in that chunk.** This is the check that matters. A
   model can attach a fluent, plausible, entirely fabricated sentence to a
   genuine chunk_id, and check 1 alone would wave it through -- which is the
   precise failure this project exists to prevent.

Whitespace is normalised on both sides before comparing. That forgives
line-wrapping, which is an artifact of how PDF text was rendered, and nothing
else: no case folding, no punctuation stripping, no fuzzy matching. A quote
that differs by a word is a failed citation, not a near miss.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rbi_rag.schemas import Chunk, Citation, CitationClaim, RetrievedChunk

#: Short enough to match almost anything, so it is not evidence of anything.
MIN_QUOTE_CHARS = 12

_WHITESPACE_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse whitespace runs so line wrapping does not fail a real quote."""
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class ValidationResult:
    """Citations that survived, and why the others did not."""

    citations: list[Citation] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _citation_from(chunk: Chunk, quote: str) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        circular_number=chunk.circular_number,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        url=chunk.url,
        quote=quote,
    )


def validate_citations(
    claims: Sequence[CitationClaim],
    retrieved: Sequence[RetrievedChunk],
) -> ValidationResult:
    """Check every claim against what the model was actually shown."""
    by_id: dict[str, Chunk] = {item.chunk.chunk_id: item.chunk for item in retrieved}
    normalised_text = {chunk_id: normalise(chunk.text) for chunk_id, chunk in by_id.items()}

    citations: list[Citation] = []
    violations: list[str] = []
    seen: set[tuple[str, str]] = set()

    for claim in claims:
        chunk = by_id.get(claim.chunk_id)
        if chunk is None:
            violations.append(f"chunk_id {claim.chunk_id!r} was not among the excerpts provided")
            continue

        quote = claim.quote.strip()
        if len(normalise(quote)) < MIN_QUOTE_CHARS:
            violations.append(
                f"quote for {claim.chunk_id} is too short to support anything: {quote!r}"
            )
            continue

        if normalise(quote) not in normalised_text[claim.chunk_id]:
            violations.append(
                f"quote for {claim.chunk_id} does not appear in that excerpt: {quote[:80]!r}"
            )
            continue

        key = (claim.chunk_id, normalise(quote))
        if key in seen:
            continue
        seen.add(key)
        citations.append(_citation_from(chunk, quote))

    return ValidationResult(citations=citations, violations=violations)
