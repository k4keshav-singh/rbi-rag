"""Citation validation.

The check that a quote really appears in the chunk it is attributed to is the
one thing standing between this system and a confident fabrication, so most of
these tests are about ways a wrong quote could slip through.
"""

import pytest

from rbi_rag.generate.validate import MIN_QUOTE_CHARS, normalise, validate_citations
from rbi_rag.schemas import CitationClaim, RetrievedChunk
from tests.fakes import CAPITAL_TEXT, TIER_ONE_TEXT, chunk


def retrieved(*chunk_ids: str) -> list[RetrievedChunk]:
    texts = {"c-capital": CAPITAL_TEXT, "c-tier1": TIER_ONE_TEXT}
    return [
        RetrievedChunk(chunk=chunk(cid, texts[cid]), score=0.9, rank=index)
        for index, cid in enumerate(chunk_ids, start=1)
    ]


def claim(chunk_id: str, quote: str) -> CitationClaim:
    return CitationClaim(chunk_id=chunk_id, quote=quote)


class TestAcceptsGenuineCitations:
    def test_an_exact_quote_passes(self) -> None:
        result = validate_citations(
            [claim("c-capital", "minimum Capital to Risk-weighted Assets Ratio")],
            retrieved("c-capital"),
        )
        assert result.ok
        assert len(result.citations) == 1

    def test_the_citation_carries_source_metadata(self) -> None:
        """A citation is only verifiable if it says where to look."""
        result = validate_citations(
            [claim("c-capital", "nine per cent on an ongoing basis")],
            retrieved("c-capital"),
        )
        citation = result.citations[0]
        assert citation.circular_number == "RBI/DOR/2023-24/106"
        assert citation.page_start == 3
        assert citation.url.endswith(".PDF")

    def test_line_wrapping_is_forgiven(self) -> None:
        """PDF text wraps; that is rendering, not a different quote."""
        result = validate_citations(
            [claim("c-capital", "minimum Capital to\n   Risk-weighted   Assets Ratio")],
            retrieved("c-capital"),
        )
        assert result.ok

    def test_several_citations_across_chunks(self) -> None:
        result = validate_citations(
            [
                claim("c-capital", "nine per cent on an ongoing basis"),
                claim("c-tier1", "seven per cent of risk-weighted assets"),
            ],
            retrieved("c-capital", "c-tier1"),
        )
        assert result.ok
        assert {c.chunk_id for c in result.citations} == {"c-capital", "c-tier1"}

    def test_duplicates_are_collapsed(self) -> None:
        quote = "nine per cent on an ongoing basis"
        result = validate_citations(
            [claim("c-capital", quote), claim("c-capital", quote)],
            retrieved("c-capital"),
        )
        assert result.ok
        assert len(result.citations) == 1


class TestRejectsFabrication:
    def test_an_unknown_chunk_id_is_rejected(self) -> None:
        result = validate_citations(
            [claim("c-invented", "nine per cent on an ongoing basis")],
            retrieved("c-capital"),
        )
        assert not result.ok
        assert "was not among the excerpts" in result.violations[0]

    def test_a_fabricated_quote_on_a_real_chunk_is_rejected(self) -> None:
        """The dangerous case: a real id, a plausible sentence, no source."""
        result = validate_citations(
            [claim("c-capital", "Banks shall maintain twelve per cent at all times.")],
            retrieved("c-capital"),
        )
        assert not result.ok
        assert "does not appear" in result.violations[0]
        assert result.citations == []

    def test_a_paraphrase_is_rejected(self) -> None:
        """Close is not quoted. No fuzzy matching, deliberately."""
        result = validate_citations(
            [claim("c-capital", "a minimum capital ratio of 9%")],
            retrieved("c-capital"),
        )
        assert not result.ok

    def test_a_quote_from_a_different_retrieved_chunk_is_rejected(self) -> None:
        """Right corpus, wrong attribution - still an unverifiable citation."""
        result = validate_citations(
            [claim("c-capital", "seven per cent of risk-weighted assets")],
            retrieved("c-capital", "c-tier1"),
        )
        assert not result.ok

    def test_case_changes_are_not_forgiven(self) -> None:
        result = validate_citations(
            [claim("c-capital", "NINE PER CENT ON AN ONGOING BASIS")],
            retrieved("c-capital"),
        )
        assert not result.ok

    def test_a_trivially_short_quote_is_rejected(self) -> None:
        """'the' appears in everything and evidences nothing."""
        result = validate_citations([claim("c-capital", "bank")], retrieved("c-capital"))
        assert not result.ok
        assert "too short" in result.violations[0]

    def test_good_and_bad_claims_are_reported_separately(self) -> None:
        result = validate_citations(
            [
                claim("c-capital", "nine per cent on an ongoing basis"),
                claim("c-capital", "and also fifteen per cent on Tuesdays"),
            ],
            retrieved("c-capital"),
        )
        assert len(result.citations) == 1
        assert len(result.violations) == 1


class TestEdges:
    def test_no_claims_is_vacuously_ok(self) -> None:
        result = validate_citations([], retrieved("c-capital"))
        assert result.ok
        assert result.citations == []

    def test_nothing_retrieved_means_nothing_can_be_cited(self) -> None:
        result = validate_citations([claim("c-capital", "nine per cent basis")], [])
        assert not result.ok

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("a  b", "a b"), ("a\nb", "a b"), ("  a b  ", "a b"), ("a\t\tb", "a b")],
    )
    def test_normalise_collapses_whitespace_only(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    def test_minimum_quote_length_is_meaningful(self) -> None:
        assert MIN_QUOTE_CHARS >= 10
