"""Parsing the RBI listing page. Fixture-driven; never touches the network."""

from datetime import date

from rbi_rag.ingest.harvest import doc_id_from_detail_url, parse_header_date, parse_listing

EN_DASH = "\u2013"


def test_parses_every_complete_row(listing_html: str) -> None:
    entries, without_pdf = parse_listing(listing_html)
    assert [e.doc_id for e in entries] == ["11322", "10476", "13443"]
    assert without_pdf == 1


def test_entries_inherit_the_nearest_category_and_date(listing_html: str) -> None:
    entries = {e.doc_id: e for e in parse_listing(listing_html)[0]}

    assert entries["11322"].category == "Banker and Debt Manager to Government"
    assert entries["11322"].published_on == date(2018, 7, 3)

    # Second date header under the same category.
    assert entries["10476"].category == "Banker and Debt Manager to Government"
    assert entries["10476"].published_on == date(2016, 7, 1)

    # New category heading, new date beneath it.
    assert entries["13443"].category == "Commercial Banks"
    assert entries["13443"].published_on == date(2026, 10, 7)


def test_relative_detail_links_are_absolutised(listing_html: str) -> None:
    entries = {e.doc_id: e for e in parse_listing(listing_html)[0]}
    assert entries["11322"].detail_url.startswith("https://www.rbi.org.in/Scripts/")
    assert entries["11322"].pdf_url == "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/61MD.PDF"


def test_html_entities_in_titles_are_decoded(listing_html: str) -> None:
    entries = {e.doc_id: e for e in parse_listing(listing_html)[0]}
    # The source writes an en dash as &#8211;; a mangled title would corrupt the manifest.
    assert EN_DASH in entries["10476"].title
    assert "&#" not in entries["10476"].title


def test_rows_without_a_pdf_are_counted_not_dropped_silently(listing_html: str) -> None:
    entries, without_pdf = parse_listing(listing_html)
    assert "13444" not in {e.doc_id for e in entries}
    assert without_pdf == 1


def test_empty_page_yields_nothing() -> None:
    entries, without_pdf = parse_listing("<html><body></body></html>")
    assert entries == []
    assert without_pdf == 0


class TestParseHeaderDate:
    def test_accepts_the_listing_format(self) -> None:
        assert parse_header_date("Jul 03, 2018") == date(2018, 7, 3)

    def test_accepts_an_unpadded_day(self) -> None:
        assert parse_header_date("Jul 3, 2018") == date(2018, 7, 3)

    def test_rejects_a_category_heading(self) -> None:
        assert parse_header_date("Commercial Banks") is None

    def test_rejects_an_impossible_date(self) -> None:
        assert parse_header_date("Feb 30, 2018") is None


class TestDocIdFromDetailUrl:
    def test_extracts_the_id(self) -> None:
        assert doc_id_from_detail_url("BS_ViewMasDirections.aspx?id=11322") == "11322"

    def test_returns_none_without_an_id(self) -> None:
        assert doc_id_from_detail_url("BS_ViewMasDirections.aspx") is None
