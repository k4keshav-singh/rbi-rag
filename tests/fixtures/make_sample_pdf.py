"""Generate the committed sample PDF used by the ingest tests.

Run with ``uv run python tests/fixtures/make_sample_pdf.py`` to regenerate.

Writing our own fixture rather than committing a real RBI circular keeps the
repository free of third-party content and lets the fixture contain exactly the
features the cleaner and chunker need to be tested against: a running header
and footer on every page, a word hyphenated across a line break, numbered
headings, and a first page carrying circular identifiers.

Paragraphs are laid out with ``insert_textbox``, one box per paragraph, because
that is how a real PDF is structured: a paragraph is a single text block whose
lines are separated by newlines. Drawing line by line instead would make every
line its own block and quietly invalidate the chunking and cleaning tests.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

HEADER = "RESERVE BANK OF INDIA"
FOOTER_TEMPLATE = "Page {page} of {total}"

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
LEFT = 72.0
RIGHT = PAGE_WIDTH - 72.0
FIRST_BASELINE = 110.0
LINE_HEIGHT = 15.0
PARAGRAPH_GAP = 10.0
FONT_SIZE = 11.0

#: Each page is a list of paragraphs. A newline inside a paragraph is a real
#: line break within one block, which is what the hyphenation rule must handle.
PAGES: list[list[str]] = [
    [
        "RBI/DOR/2023-24/106",
        "DOR.FIN.REC.No.45/03.10.123/2023-24",
        "Master Direction - Illustrative Prudential Norms",
        "1. Introduction",
        "These Directions are issued in exercise of the powers conferred\n"
        "under the Banking Regulation Act. They apply to every commercial\n"
        "bank and set out the minimum prudential requirements that apply.",
        "This paragraph contains a word split across a line break by hyphena-\n"
        "tion, so that the cleaner can be tested on rejoining it correctly.",
    ],
    [
        "2. Definitions",
        "In these Directions, unless the context otherwise requires, the\n"
        "terms below bear the meanings assigned to them here.",
        "Capital to Risk-weighted Assets Ratio means the ratio of eligible\n"
        "capital to aggregate risk-weighted assets, as a percentage.",
        "Eligible capital comprises Tier 1 capital and Tier 2 capital, each\n"
        "computed in the manner specified in these Directions.",
    ],
    [
        "3. Capital Requirements",
        "Every bank shall maintain a minimum Capital to Risk-weighted\n"
        "Assets Ratio of nine per cent on an ongoing basis.",
        "3.1 Tier 1 Capital",
        "Tier 1 capital shall not be less than seven per cent of risk-\n"
        "weighted assets at any time during the financial year.",
        "3.2 Tier 2 Capital",
        "Tier 2 capital shall not exceed Tier 1 capital for the purpose of\n"
        "computing the overall capital adequacy ratio.",
    ],
    [
        "4. Reporting",
        "Every bank shall report its capital position to the Reserve Bank\n"
        "within fifteen days of the close of each quarter.",
        "Non-compliance with these Directions shall attract such action as\n"
        "the Reserve Bank may consider appropriate.",
    ],
]


def build() -> pymupdf.Document:
    document = pymupdf.open()
    total = len(PAGES)

    for number, paragraphs in enumerate(PAGES, start=1):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text((LEFT, 60), HEADER, fontsize=9, fontname="helv")

        top = FIRST_BASELINE
        for paragraph in paragraphs:
            lines = paragraph.count("\n") + 1
            height = lines * LINE_HEIGHT + 6.0
            rect = pymupdf.Rect(LEFT, top, RIGHT, top + height)
            overflow = page.insert_textbox(
                rect, paragraph, fontsize=FONT_SIZE, fontname="helv", lineheight=1.3
            )
            if overflow < 0:
                raise RuntimeError(f"paragraph did not fit on page {number}: {paragraph[:40]!r}")
            top += height + PARAGRAPH_GAP

        page.insert_text(
            (LEFT, PAGE_HEIGHT - 62),
            FOOTER_TEMPLATE.format(page=number, total=total),
            fontsize=9,
            fontname="helv",
        )

    document.set_toc(
        [
            [1, "1. Introduction", 1],
            [1, "2. Definitions", 2],
            [1, "3. Capital Requirements", 3],
            [1, "4. Reporting", 4],
        ]
    )
    return document


def main() -> None:
    destination = Path(__file__).parent / "sample.pdf"
    document = build()
    # deflate=True keeps the committed file small; garbage=4 keeps it stable.
    document.save(destination, deflate=True, garbage=4)
    document.close()
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
