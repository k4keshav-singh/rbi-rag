"""Cleaning rules.

Each rule can silently corrupt a citation by over-reaching, so the tests pin
down what each one must *not* touch as much as what it must remove.
"""

from rbi_rag.ingest.clean import (
    clean_pages,
    dehyphenate,
    find_repeated_lines,
    is_page_furniture,
    normalise_whitespace,
    strip_lines,
)
from rbi_rag.ingest.parse import extract_pages
from rbi_rag.schemas import PageText


def pages(*texts: str) -> list[PageText]:
    return [PageText(page_number=i, text=t) for i, t in enumerate(texts, start=1)]


class TestPageFurniture:
    def test_detects_a_bare_number(self) -> None:
        assert is_page_furniture("  7  ")

    def test_detects_page_x_of_y(self) -> None:
        assert is_page_furniture("Page 3 of 48")

    def test_leaves_a_numbered_heading_alone(self) -> None:
        assert not is_page_furniture("3. Capital Requirements")

    def test_leaves_prose_alone(self) -> None:
        assert not is_page_furniture("nine per cent")


class TestFindRepeatedLines:
    def test_finds_a_running_header(self) -> None:
        body = "RESERVE BANK OF INDIA\nreal content for this page\nmore"
        found = find_repeated_lines(pages(body, body, body, body))
        assert "RESERVE BANK OF INDIA" in found

    def test_ignores_a_phrase_repeated_in_the_body(self) -> None:
        """Only the first and last few lines of a page are candidates.

        A phrase like "shall apply" recurs throughout a regulatory document.
        Treating it as furniture would delete real text from every page.
        """
        documents = [
            PageText(
                page_number=number,
                text="\n".join(
                    [
                        f"opening line {number}",
                        f"second line {number}",
                        f"third line {number}",
                        "shall apply",
                        f"antepenultimate {number}",
                        f"penultimate {number}",
                        f"closing line {number}",
                    ]
                ),
            )
            for number in range(1, 6)
        ]
        assert "shall apply" not in find_repeated_lines(documents)

    def test_does_nothing_for_a_short_document(self) -> None:
        assert find_repeated_lines(pages("a\nb", "a\nb")) == set()


class TestStripLines:
    def test_removes_the_named_lines_and_furniture(self) -> None:
        text = "HEADER\nreal content\n12"
        assert strip_lines(text, {"HEADER"}) == "real content"

    def test_keeps_everything_else(self) -> None:
        assert strip_lines("alpha\nbeta", set()) == "alpha\nbeta"


class TestDehyphenate:
    def test_rejoins_a_split_word(self) -> None:
        assert dehyphenate("regula-\ntion applies") == "regulation applies"

    def test_leaves_a_real_compound_alone(self) -> None:
        """A capital after the break signals a compound, not a split word."""
        assert dehyphenate("cross-\nBorder") == "cross-\nBorder"

    def test_leaves_an_unbroken_hyphen_alone(self) -> None:
        assert dehyphenate("risk-weighted assets") == "risk-weighted assets"


class TestNormaliseWhitespace:
    def test_collapses_long_blank_runs_to_one(self) -> None:
        assert normalise_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_a_single_paragraph_break(self) -> None:
        assert normalise_whitespace("a\n\nb") == "a\n\nb"

    def test_strips_trailing_spaces(self) -> None:
        assert normalise_whitespace("a   \nb") == "a\nb"

    def test_normalises_windows_line_endings(self) -> None:
        assert normalise_whitespace("a\r\nb") == "a\nb"


class TestCleanPagesEndToEnd:
    def test_removes_furniture_and_rejoins_hyphens(self, sample_pdf_bytes: bytes) -> None:
        cleaned = clean_pages(extract_pages(sample_pdf_bytes))
        combined = "\n".join(page.text for page in cleaned)
        assert "RESERVE BANK OF INDIA" not in combined
        assert "Page 1 of 4" not in combined
        assert "hyphenation" in combined
        assert "hyphena-" not in combined

    def test_keeps_the_substance(self, sample_pdf_bytes: bytes) -> None:
        cleaned = clean_pages(extract_pages(sample_pdf_bytes))
        combined = "\n".join(page.text for page in cleaned)
        assert "nine per cent" in combined
        assert "RBI/DOR/2023-24/106" in combined

    def test_page_numbers_are_preserved(self, sample_pdf_bytes: bytes) -> None:
        cleaned = clean_pages(extract_pages(sample_pdf_bytes))
        assert [p.page_number for p in cleaned] == [1, 2, 3, 4]
