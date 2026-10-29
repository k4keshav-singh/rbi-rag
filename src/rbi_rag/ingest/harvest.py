"""Harvest Master Direction metadata from the RBI listing page.

The listing carries everything the manifest needs except the circular number
and department, which RBI prints on the first page of each PDF. Reading them
from the PDF instead of from a per-document detail page turns ~380 requests
into one, which is both faster and considerably politer.

The page is a single table whose rows are either headers or entries::

    <tr><td class="tableheader"><b>Commercial Banks</b></td></tr>   <- category
    <tr><td class="tableheader"><b>Oct 07, 2026</b></td></tr>       <- date
    <tr><td><a class="link2" href=...id=11322>Title</a></td>
        <td><a id='APDF_...' href='https://rbidocs.../X.PDF'></a></td></tr>

Entries inherit the most recent category and the most recent date above them.
"""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

from rbi_rag.schemas import ManifestEntry

LISTING_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

#: "Jul 03, 2018" and "Jul 3, 2018" both occur.
_DATE_RE = re.compile(r"^([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})$")


def parse_header_date(text: str) -> date | None:
    """Parse a listing date header, or return None if it is a category heading."""
    match = _DATE_RE.match(text.strip())
    if match is None:
        return None
    month = _MONTHS.get(match.group(1)[:3].lower())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def doc_id_from_detail_url(url: str) -> str | None:
    """Pull RBI's own ``id`` query parameter out of a detail-page URL."""
    values = parse_qs(urlparse(url).query).get("id")
    if not values or not values[0].strip():
        return None
    return values[0].strip()


class _ListingParser(HTMLParser):
    """Walks the listing table, emitting one entry per complete row."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self.entries: list[ManifestEntry] = []
        self.rows_without_pdf = 0

        self._category: str | None = None
        self._date: date | None = None

        self._in_header = False
        self._header_parts: list[str] = []

        self._in_title = False
        self._title_parts: list[str] = []
        self._detail_url: str | None = None
        self._pdf_url: str | None = None

    # -- tags -------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: (value or "") for name, value in attrs}
        if tag == "tr":
            self._reset_row()
        elif tag == "td" and "tableheader" in attributes.get("class", ""):
            self._in_header = True
            self._header_parts = []
        elif tag == "a":
            self._handle_anchor(attributes)

    def _handle_anchor(self, attributes: dict[str, str]) -> None:
        # Some hrefs on the listing carry trailing whitespace, which would end
        # up inside the stored URL and break equality checks on re-harvest.
        href = attributes.get("href", "").strip()
        if "link2" in attributes.get("class", "") and "id=" in href:
            self._in_title = True
            self._title_parts = []
            self._detail_url = urljoin(self._base_url, href)
        elif href.upper().endswith(".PDF"):
            self._pdf_url = urljoin(self._base_url, href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_header:
            self._in_header = False
            self._consume_header("".join(self._header_parts).strip())
        elif tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "tr":
            self._emit_row()

    def handle_data(self, data: str) -> None:
        if self._in_header:
            self._header_parts.append(data)
        elif self._in_title:
            self._title_parts.append(data)

    # -- row handling -----------------------------------------------------

    def _consume_header(self, text: str) -> None:
        if not text:
            return
        parsed = parse_header_date(text)
        if parsed is not None:
            self._date = parsed
        else:
            self._category = text
            # A new category restarts the date grouping beneath it.
            self._date = None

    def _reset_row(self) -> None:
        self._title_parts = []
        self._detail_url = None
        self._pdf_url = None
        self._in_title = False

    def _emit_row(self) -> None:
        title = " ".join("".join(self._title_parts).split())
        if not title or self._detail_url is None:
            self._reset_row()
            return
        if self._pdf_url is None:
            self.rows_without_pdf += 1
            self._reset_row()
            return
        doc_id = doc_id_from_detail_url(self._detail_url)
        if doc_id is None:
            self.rows_without_pdf += 1
            self._reset_row()
            return

        self.entries.append(
            ManifestEntry(
                doc_id=doc_id,
                title=title,
                detail_url=self._detail_url,
                pdf_url=self._pdf_url,
                category=self._category,
                published_on=self._date,
            )
        )
        self._reset_row()


def parse_listing(html: str, base_url: str = LISTING_URL) -> tuple[list[ManifestEntry], int]:
    """Parse the listing page into manifest entries.

    Returns the entries plus the number of rows that looked like entries but
    carried no usable PDF link, so a caller can report coverage honestly rather
    than silently dropping them.
    """
    parser = _ListingParser(base_url)
    parser.feed(html)
    parser.close()

    deduplicated: dict[str, ManifestEntry] = {}
    for entry in parser.entries:
        deduplicated.setdefault(entry.doc_id, entry)
    return list(deduplicated.values()), parser.rows_without_pdf
