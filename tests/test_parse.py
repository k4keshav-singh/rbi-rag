"""PDF parsing against the committed fixture."""

import pytest

from rbi_rag.ingest.parse import extract_circular_metadata, extract_outline, extract_pages


def test_extracts_every_page_one_indexed(sample_pdf_bytes: bytes) -> None:
    pages = extract_pages(sample_pdf_bytes)
    assert len(pages) == 4
    assert [p.page_number for p in pages] == [1, 2, 3, 4]


def test_paragraphs_are_separated_by_blank_lines(sample_pdf_bytes: bytes) -> None:
    """The chunker packs paragraphs, so losing the boundaries breaks chunking."""
    pages = extract_pages(sample_pdf_bytes)
    assert "\n\n" in pages[0].text


def test_page_text_matches_the_page_it_came_from(sample_pdf_bytes: bytes) -> None:
    pages = extract_pages(sample_pdf_bytes)
    assert "1. Introduction" in pages[0].text
    assert "2. Definitions" in pages[1].text
    assert "3. Capital Requirements" in pages[2].text
    assert "4. Reporting" in pages[3].text


def test_reads_the_outline(sample_pdf_bytes: bytes) -> None:
    outline = extract_outline(sample_pdf_bytes)
    assert [(item.title, item.page_number) for item in outline] == [
        ("1. Introduction", 1),
        ("2. Definitions", 2),
        ("3. Capital Requirements", 3),
        ("4. Reporting", 4),
    ]


def test_a_pdf_without_an_outline_yields_an_empty_list() -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    assert extract_outline(data) == []


class TestCircularMetadata:
    def test_reads_the_identifiers_off_the_fixture(self, sample_pdf_bytes: bytes) -> None:
        pages = extract_pages(sample_pdf_bytes)
        metadata = extract_circular_metadata(pages[0].text)
        assert metadata.circular_number == "RBI/DOR/2023-24/106"
        assert metadata.department == "DOR"
        assert metadata.department_reference == "DOR.FIN.REC.No.45/03.10.123/2023-24"

    def test_reads_the_older_format_without_a_department_segment(self) -> None:
        text = "RBI/2015-16/23\nDBR.No.BP.BC.1/21.06.201/2015-16\nJuly 1, 2015"
        metadata = extract_circular_metadata(text)
        assert metadata.circular_number == "RBI/2015-16/23"
        assert metadata.department == "DBR"

    def test_falls_back_to_the_department_inside_the_master_number(self) -> None:
        metadata = extract_circular_metadata("RBI/DPSS/2021-22/70 and nothing else")
        assert metadata.circular_number == "RBI/DPSS/2021-22/70"
        assert metadata.department == "DPSS"

    def test_returns_none_rather_than_guessing(self) -> None:
        metadata = extract_circular_metadata("A page with no identifiers on it at all.")
        assert metadata.circular_number is None
        assert metadata.department is None
        assert metadata.department_reference is None
