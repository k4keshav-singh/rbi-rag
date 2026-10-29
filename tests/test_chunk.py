"""Chunking.

Chunk size is the independent variable of the Phase 4 ablation, so the budget
has to actually hold; and every chunk carries the citation metadata, so page
attribution has to be exact.
"""

import pytest

from rbi_rag.ingest.chunk import (
    TextUnit,
    assign_sections,
    chunk_document,
    iter_units,
    looks_like_heading,
    pack,
    split_oversized,
)
from rbi_rag.ingest.clean import clean_pages
from rbi_rag.ingest.parse import OutlineItem, extract_outline, extract_pages
from rbi_rag.ingest.tokenize import WhitespaceTokenCounter
from rbi_rag.schemas import PageText

COUNTER = WhitespaceTokenCounter()


def words(n: int, token: str = "word") -> str:
    return " ".join([token] * n)


def page(number: int, *paragraphs: str) -> PageText:
    return PageText(page_number=number, text="\n\n".join(paragraphs))


class TestIterUnits:
    def test_splits_on_blank_lines_and_keeps_the_page(self) -> None:
        units = iter_units([page(1, "first para", "second para"), page(2, "third")])
        assert [(u.text, u.page_number) for u in units] == [
            ("first para", 1),
            ("second para", 1),
            ("third", 2),
        ]

    def test_collapses_internal_whitespace(self) -> None:
        units = iter_units([page(1, "a\nb   c")])
        assert units[0].text == "a b c"

    def test_drops_empty_blocks(self) -> None:
        assert iter_units([page(1, "", "   ", "real")]) == [TextUnit("real", 1)]


class TestLooksLikeHeading:
    @pytest.mark.parametrize(
        "line",
        ["1. Introduction", "3.1 Tier 1 Capital", "CHAPTER II", "DEFINITIONS AND SCOPE"],
    )
    def test_accepts_headings(self, line: str) -> None:
        assert looks_like_heading(line)

    @pytest.mark.parametrize(
        "line",
        [
            "Every bank shall maintain a minimum ratio of nine per cent.",
            "",
            "the quick brown fox jumps over the lazy dog and keeps on running far away now",
        ],
    )
    def test_rejects_prose(self, line: str) -> None:
        assert not looks_like_heading(line)


class TestSplitOversized:
    def test_leaves_a_fitting_unit_whole(self) -> None:
        unit = TextUnit(words(10), 1)
        assert split_oversized(unit, COUNTER, 20) == [unit]

    def test_splits_on_sentences_first(self) -> None:
        unit = TextUnit(f"{words(8)}. {words(8)}.", 1)
        pieces = split_oversized(unit, COUNTER, 10)
        assert len(pieces) == 2
        assert all(COUNTER.count(p.text) <= 10 for p in pieces)

    def test_splits_a_single_long_sentence_on_words(self) -> None:
        pieces = split_oversized(TextUnit(words(25), 1), COUNTER, 10)
        assert len(pieces) == 3
        assert all(COUNTER.count(p.text) <= 10 for p in pieces)

    def test_keeps_the_page_number(self) -> None:
        pieces = split_oversized(TextUnit(words(25), 7), COUNTER, 10)
        assert {p.page_number for p in pieces} == {7}


class TestPack:
    def test_respects_the_budget(self) -> None:
        units = [TextUnit(words(6), 1) for _ in range(5)]
        groups = pack(units, COUNTER, target_tokens=20, overlap_tokens=0)
        assert all(sum(COUNTER.count(u.text) for u in g) <= 20 for g in groups)

    def test_overlap_repeats_trailing_units(self) -> None:
        units = [TextUnit(f"para{i} {words(5)}", 1) for i in range(4)]
        with_overlap = pack(units, COUNTER, target_tokens=12, overlap_tokens=6)
        flattened = [u.text for g in with_overlap for u in g]
        assert len(flattened) > len(units)

    def test_zero_overlap_repeats_nothing(self) -> None:
        units = [TextUnit(f"para{i} {words(5)}", 1) for i in range(4)]
        groups = pack(units, COUNTER, target_tokens=12, overlap_tokens=0)
        flattened = [u.text for g in groups for u in g]
        assert len(flattened) == len(set(flattened)) == len(units)


class TestAssignSections:
    def test_prefers_the_outline(self) -> None:
        units = [TextUnit("alpha", 1), TextUnit("beta", 2)]
        outline = [OutlineItem(1, "First", 1), OutlineItem(1, "Second", 2)]
        tagged = assign_sections(units, outline)
        assert [u.section_title for u in tagged] == ["First", "Second"]

    def test_falls_back_to_detected_headings(self) -> None:
        units = [
            TextUnit("1. Introduction", 1),
            TextUnit("Some prose here.", 1),
            TextUnit("2. Definitions", 2),
            TextUnit("More prose.", 2),
        ]
        tagged = assign_sections(units)
        assert [u.section_title for u in tagged] == [
            "1. Introduction",
            "1. Introduction",
            "2. Definitions",
            "2. Definitions",
        ]

    def test_text_before_any_heading_has_no_section(self) -> None:
        tagged = assign_sections([TextUnit("front matter", 1)])
        assert tagged[0].section_title is None


class TestChunkDocument:
    def test_rejects_an_impossible_budget(self) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            chunk_document(
                doc_id="d",
                url="u",
                pages=[page(1, "x")],
                strategy="fixed_tokens",
                target_tokens=0,
                counter=COUNTER,
            )

    def test_rejects_an_impossible_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap_ratio"):
            chunk_document(
                doc_id="d",
                url="u",
                pages=[page(1, "x")],
                strategy="fixed_tokens",
                target_tokens=10,
                counter=COUNTER,
                overlap_ratio=1.0,
            )

    def test_an_empty_document_yields_no_chunks(self) -> None:
        assert (
            chunk_document(
                doc_id="d",
                url="u",
                pages=[],
                strategy="fixed_tokens",
                target_tokens=10,
                counter=COUNTER,
            )
            == []
        )

    def test_no_chunk_exceeds_the_budget(self, sample_pdf_bytes: bytes) -> None:
        """A chunk over the budget would be silently truncated when embedded."""
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        for size in (20, 40, 100):
            chunks = chunk_document(
                doc_id="d",
                url="u",
                pages=pages,
                strategy="fixed_tokens",
                target_tokens=size,
                counter=COUNTER,
            )
            assert chunks
            assert all((c.token_count or 0) <= size for c in chunks)

    def test_page_ranges_are_ordered_and_within_the_document(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        chunks = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=40,
            counter=COUNTER,
        )
        for chunk in chunks:
            assert 1 <= chunk.page_start <= chunk.page_end <= len(pages)

    def test_chunk_text_is_a_verbatim_substring_of_the_source(
        self, sample_pdf_bytes: bytes
    ) -> None:
        """Citations quote chunk text, so it must not be re-encoded or reflowed."""
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        source = " ".join(" ".join(p.text.split()) for p in pages)
        chunks = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=40,
            counter=COUNTER,
        )
        for chunk in chunks:
            for paragraph in chunk.text.split("\n\n"):
                assert paragraph in source

    def test_chunk_ids_are_deterministic(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        kwargs = {
            "doc_id": "d",
            "url": "u",
            "pages": pages,
            "strategy": "fixed_tokens",
            "target_tokens": 40,
            "counter": COUNTER,
        }
        first = chunk_document(**kwargs)  # type: ignore[arg-type]
        second = chunk_document(**kwargs)  # type: ignore[arg-type]
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_chunk_ids_are_unique(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        chunks = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=40,
            counter=COUNTER,
        )
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_changing_the_size_changes_the_ids(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        small = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=20,
            counter=COUNTER,
        )
        large = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=80,
            counter=COUNTER,
        )
        assert not {c.chunk_id for c in small} & {c.chunk_id for c in large}

    def test_citation_metadata_rides_on_every_chunk(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        chunks = chunk_document(
            doc_id="11322",
            url="https://example.test/x.PDF",
            pages=pages,
            strategy="fixed_tokens",
            target_tokens=40,
            counter=COUNTER,
            circular_number="RBI/DOR/2023-24/106",
        )
        assert all(c.doc_id == "11322" for c in chunks)
        assert all(c.url == "https://example.test/x.PDF" for c in chunks)
        assert all(c.circular_number == "RBI/DOR/2023-24/106" for c in chunks)

    def test_section_aware_never_mixes_sections(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        outline = extract_outline(sample_pdf_bytes)
        chunks = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="section_aware",
            target_tokens=40,
            counter=COUNTER,
            outline=outline,
        )
        titles = {c.section_title for c in chunks}
        assert titles <= {item.title for item in outline}
        assert len(titles) > 1

    def test_section_aware_records_the_strategy(self, sample_pdf_bytes: bytes) -> None:
        pages = clean_pages(extract_pages(sample_pdf_bytes))
        chunks = chunk_document(
            doc_id="d",
            url="u",
            pages=pages,
            strategy="section_aware",
            target_tokens=40,
            counter=COUNTER,
        )
        assert all(c.strategy == "section_aware" for c in chunks)
        assert all(c.target_tokens == 40 for c in chunks)
