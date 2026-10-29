"""Turn PDF bytes into per-page text, an outline, and header metadata.

Page numbers are one-indexed throughout so that a citation points at the page a
reader sees in a PDF viewer, not at an array offset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from rbi_rag.schemas import PageText

#: "RBI/DOR/2023-24/106" and the older "RBI/2015-16/23" both occur.
#: PyMuPDF block tuples are (x0, y0, x1, y1, text, block_no, block_type);
#: block_type 0 is text and 1 is an image.
_TEXT_BLOCK = 0

_MASTER_NUMBER_RE = re.compile(r"\bRBI/(?:[A-Z]{2,8}/)?\d{4}-\d{2}/\d+[A-Za-z]?\b")

#: "DOR.FIN.REC.No.45/03.10.123/2023-24", "DBR.No.BP.BC.1/21.06.201/2015-16".
_DEPARTMENT_REF_RE = re.compile(
    r"\b(?P<dept>[A-Z]{2,6})(?P<rest>(?:\.[A-Z]{2,6}){0,4}\.No\.[A-Za-z0-9./()-]+)"
)


@dataclass(frozen=True)
class OutlineItem:
    """One bookmark from the PDF's table of contents."""

    level: int
    title: str
    page_number: int


@dataclass(frozen=True)
class CircularMetadata:
    """Identifiers printed on the first page of a Master Direction."""

    circular_number: str | None
    department: str | None
    department_reference: str | None


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Extract text for every page, in reading order.

    Uses PyMuPDF's block extraction rather than flat text. Blocks correspond to
    paragraphs, and joining them with a blank line is what gives the chunker
    real paragraph boundaries to pack. Flat text extraction loses them: it
    separates every line with a single newline, so a whole page collapses into
    one paragraph and chunking degenerates into fixed-width slicing.
    """
    pages: list[PageText] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for index, page in enumerate(document, start=1):
            blocks = [b for b in page.get_text("blocks") if int(b[6]) == _TEXT_BLOCK]
            blocks.sort(key=lambda b: (round(float(b[1]), 1), float(b[0])))
            paragraphs = [str(b[4]).strip() for b in blocks]
            text = "\n\n".join(p for p in paragraphs if p)
            pages.append(PageText(page_number=index, text=text))
    return pages


def extract_outline(pdf_bytes: bytes) -> list[OutlineItem]:
    """Read the PDF bookmark tree.

    Many RBI PDFs carry no bookmarks at all, in which case this returns an empty
    list and the section-aware chunker falls back to heading detection.
    """
    items: list[OutlineItem] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for entry in document.get_toc(simple=True):
            level, title, page_number = entry[0], entry[1], entry[2]
            if page_number < 1:
                continue
            cleaned = " ".join(str(title).split())
            if cleaned:
                items.append(
                    OutlineItem(level=int(level), title=cleaned, page_number=int(page_number))
                )
    return items


def extract_circular_metadata(first_page_text: str) -> CircularMetadata:
    """Pull the circular number and issuing department off the first page.

    Returns Nones rather than guessing when the patterns do not match. Phase 4's
    citation checks depend on this being accurate, so a miss must be visible.
    """
    master_match = _MASTER_NUMBER_RE.search(first_page_text)
    circular_number = master_match.group(0) if master_match else None

    department: str | None = None
    department_reference: str | None = None
    reference_match = _DEPARTMENT_REF_RE.search(first_page_text)
    if reference_match is not None:
        department = reference_match.group("dept")
        department_reference = reference_match.group(0)
    elif master_match is not None:
        # "RBI/DOR/2023-24/106" embeds the department in the middle segment.
        segments = master_match.group(0).split("/")
        if len(segments) == 4 and segments[1].isalpha():
            department = segments[1]

    return CircularMetadata(
        circular_number=circular_number,
        department=department,
        department_reference=department_reference,
    )
