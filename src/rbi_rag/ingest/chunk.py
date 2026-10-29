"""Split a parsed document into retrievable chunks.

Two strategies, both of which keep every chunk's page range exact so a citation
can be checked against the PDF:

``fixed_tokens``
    Pack paragraphs up to a token budget, with overlap, ignoring structure.

``section_aware``
    The same packing, but never across a section boundary, so a chunk is always
    about one thing. Sections come from the PDF outline when it has one, and
    from heading detection when it does not.

Chunks are built by accumulating whole paragraphs rather than by slicing a token
array and decoding it back. Decoding is lossy -- it normalises whitespace and
can mangle punctuation -- and a citation system cannot afford chunk text that
differs from the source it points at.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rbi_rag.ingest.parse import OutlineItem
from rbi_rag.ingest.tokenize import TokenCounter
from rbi_rag.schemas import Chunk, ChunkStrategy, PageText

#: Split a paragraph into sentences when it alone exceeds the budget.
_SENTENCE_RE = re.compile(r"(?<=[.;:?!])\s+")

#: Headings used when the PDF carries no outline. RBI Master Directions are
#: mostly numbered ("3. Definitions", "3.1 Scope") or use chapter headings.
_CHAPTER_RE = re.compile(r"^(?:CHAPTER|PART|SECTION)\s+[IVXLC0-9]+\b", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*\.?\s+\S")
_SHOUTED_RE = re.compile(r"^[A-Z][A-Z \t,&/()-]{6,}$")
_HEADING_PATTERNS = (_CHAPTER_RE, _NUMBERED_RE, _SHOUTED_RE)

#: A bare section number such as "3.1." may end in a full stop; prose may not.
_BARE_NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*\.$")

_MAX_HEADING_WORDS = 20


@dataclass(frozen=True)
class TextUnit:
    """A paragraph, tagged with the page it came from."""

    text: str
    page_number: int
    section_title: str | None = None


def iter_units(pages: Sequence[PageText]) -> list[TextUnit]:
    """Split pages into paragraphs, preserving page attribution."""
    units: list[TextUnit] = []
    for page in pages:
        for block in page.text.split("\n\n"):
            text = " ".join(block.split())
            if text:
                units.append(TextUnit(text=text, page_number=page.page_number))
    return units


def looks_like_heading(line: str) -> bool:
    """True for lines that read as a section heading rather than prose."""
    stripped = line.strip()
    if not stripped or len(stripped.split()) > _MAX_HEADING_WORDS:
        return False
    if stripped.endswith(".") and not _BARE_NUMBER_RE.match(stripped):
        return False
    return any(pattern.match(stripped) for pattern in _HEADING_PATTERNS)


def assign_sections(
    units: Sequence[TextUnit],
    outline: Sequence[OutlineItem] = (),
) -> list[TextUnit]:
    """Tag each unit with the section it belongs to.

    Prefers the PDF outline, which is authoritative where it exists. Falls back
    to heading detection, which is a heuristic and is documented as one.
    """
    if outline:
        ordered = sorted(outline, key=lambda item: item.page_number)
        from_outline: list[TextUnit] = []
        for unit in units:
            title: str | None = None
            for item in ordered:
                if item.page_number > unit.page_number:
                    break
                title = item.title
            from_outline.append(
                TextUnit(text=unit.text, page_number=unit.page_number, section_title=title)
            )
        return from_outline

    tagged: list[TextUnit] = []
    current: str | None = None
    for unit in units:
        if looks_like_heading(unit.text):
            current = unit.text
        tagged.append(TextUnit(text=unit.text, page_number=unit.page_number, section_title=current))
    return tagged


def split_oversized(unit: TextUnit, counter: TokenCounter, target_tokens: int) -> list[TextUnit]:
    """Break up a paragraph that exceeds the budget on its own.

    Sentences first; if a single sentence is still too long, split on words. A
    chunk larger than the embedding model's window is silently truncated at
    embed time, so this never gives up and emits an oversized unit.
    """
    if counter.count(unit.text) <= target_tokens:
        return [unit]

    pieces: list[str] = []
    for sentence in _SENTENCE_RE.split(unit.text):
        candidate_sentence = sentence.strip()
        if not candidate_sentence:
            continue
        if counter.count(candidate_sentence) <= target_tokens:
            pieces.append(candidate_sentence)
            continue
        buffer: list[str] = []
        for word in candidate_sentence.split():
            grown = [*buffer, word]
            if buffer and counter.count(" ".join(grown)) > target_tokens:
                pieces.append(" ".join(buffer))
                buffer = [word]
            else:
                buffer = grown
        if buffer:
            pieces.append(" ".join(buffer))

    return [
        TextUnit(text=piece, page_number=unit.page_number, section_title=unit.section_title)
        for piece in pieces
        if piece
    ]


def _carry_overlap(
    group: Sequence[TextUnit],
    counter: TokenCounter,
    overlap_tokens: int,
) -> tuple[list[TextUnit], int]:
    """Take trailing units from a finished group to seed the next one."""
    if overlap_tokens <= 0:
        return [], 0
    carried: list[TextUnit] = []
    total = 0
    for unit in reversed(group):
        tokens = counter.count(unit.text)
        if total + tokens > overlap_tokens:
            break
        carried.insert(0, unit)
        total += tokens
    return carried, total


def pack(
    units: Sequence[TextUnit],
    counter: TokenCounter,
    target_tokens: int,
    overlap_tokens: int,
) -> list[list[TextUnit]]:
    """Greedily pack units into groups of at most ``target_tokens``."""
    groups: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    current_tokens = 0

    for unit in units:
        for piece in split_oversized(unit, counter, target_tokens):
            piece_tokens = counter.count(piece.text)
            if current and current_tokens + piece_tokens > target_tokens:
                groups.append(current)
                current, current_tokens = _carry_overlap(current, counter, overlap_tokens)
                if current_tokens + piece_tokens > target_tokens:
                    # The overlap would push this chunk over budget, and an
                    # over-budget chunk is silently truncated at embed time.
                    # Context is worth less than a chunk that is actually whole.
                    current, current_tokens = [], 0
            current.append(piece)
            current_tokens += piece_tokens

    if current:
        groups.append(current)
    return groups


def _group_by_section(units: Sequence[TextUnit]) -> list[list[TextUnit]]:
    """Split a tagged unit stream wherever the section title changes."""
    groups: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    current_title: str | None = None

    for unit in units:
        if current and unit.section_title != current_title:
            groups.append(current)
            current = []
        current.append(unit)
        current_title = unit.section_title

    if current:
        groups.append(current)
    return groups


def _chunk_id(doc_id: str, strategy: str, target_tokens: int, index: int, text: str) -> str:
    """Deterministic id: identical input always yields an identical chunk id."""
    payload = f"{doc_id}|{strategy}|{target_tokens}|{index}|{text}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return f"{doc_id}-{strategy}-{target_tokens}-{index:04d}-{digest[:8]}"


def chunk_document(
    *,
    doc_id: str,
    url: str,
    pages: Sequence[PageText],
    strategy: ChunkStrategy,
    target_tokens: int,
    counter: TokenCounter,
    circular_number: str | None = None,
    outline: Sequence[OutlineItem] = (),
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Chunk one document under the given strategy and size."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be at least 1")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")

    units = iter_units(pages)
    if not units:
        return []

    overlap_tokens = int(target_tokens * overlap_ratio)

    if strategy == "fixed_tokens":
        groups = pack(units, counter, target_tokens, overlap_tokens)
    else:
        groups = []
        for section_units in _group_by_section(assign_sections(units, outline)):
            groups.extend(pack(section_units, counter, target_tokens, overlap_tokens))

    chunks: list[Chunk] = []
    for index, group in enumerate(groups):
        text = "\n\n".join(unit.text for unit in group)
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, strategy, target_tokens, index, text),
                doc_id=doc_id,
                text=text,
                page_start=min(unit.page_number for unit in group),
                page_end=max(unit.page_number for unit in group),
                circular_number=circular_number,
                url=url,
                strategy=strategy,
                target_tokens=target_tokens,
                token_count=counter.count(text),
                section_title=group[0].section_title,
            )
        )
    return chunks
