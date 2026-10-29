"""Clean extracted PDF text.

Each rule is a separate function because each one can silently corrupt a
citation if it over-reaches: a header rule that eats a real paragraph, or a
de-hyphenation rule that welds two words together, both produce text that looks
fine and quotes wrongly.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from rbi_rag.schemas import PageText

#: Lines that are page furniture regardless of how often they repeat.
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|of)?\s*\d*\s*$", re.IGNORECASE)

#: "regula-\ntion" -> "regulation", but never "cross-\nborder" -> "crossborder"
#: when the second part starts with a capital, which signals a real compound.
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def is_page_furniture(line: str) -> bool:
    """True for bare page numbers and 'Page 3 of 48' style lines."""
    return bool(_PAGE_NUMBER_RE.match(line))


def find_repeated_lines(
    pages: Sequence[PageText],
    *,
    edge_lines: int = 3,
    min_pages: int = 3,
    min_fraction: float = 0.5,
) -> set[str]:
    """Find running headers and footers.

    Only the first and last ``edge_lines`` of each page are considered, so a
    sentence that happens to recur in the body is never mistaken for furniture.
    A line must appear on at least ``min_pages`` pages *and* on at least
    ``min_fraction`` of them to count.
    """
    if len(pages) < min_pages:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        counts.update(set(edges))

    threshold = max(min_pages, int(len(pages) * min_fraction))
    return {line for line, count in counts.items() if count >= threshold}


def strip_lines(text: str, unwanted: Iterable[str]) -> str:
    """Drop exactly the given lines, plus any page-number furniture."""
    blocked = set(unwanted)
    kept = [
        line
        for line in text.splitlines()
        if line.strip() not in blocked and not is_page_furniture(line)
    ]
    return "\n".join(kept)


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by a hyphen."""
    return _HYPHEN_BREAK_RE.sub(r"\1", text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of blank lines and strip trailing spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_pages(pages: Sequence[PageText]) -> list[PageText]:
    """Apply every cleaning rule, preserving page boundaries."""
    repeated = find_repeated_lines(pages)
    cleaned: list[PageText] = []
    for page in pages:
        text = strip_lines(page.text, repeated)
        text = dehyphenate(text)
        text = normalise_whitespace(text)
        cleaned.append(PageText(page_number=page.page_number, text=text))
    return cleaned
