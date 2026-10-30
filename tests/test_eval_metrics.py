"""Relevance and metrics.

The property that matters most is at the bottom: a gold span scores the same
whatever chunking configuration it is measured against. Without that the
ablation compares arms that were graded differently.
"""

import math

import pytest

from rbi_rag.eval.metrics import (
    abstention_rates,
    citation_precision,
    cohens_kappa,
    mean_ignoring_none,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rbi_rag.eval.relevance import (
    chunk_covers_span,
    novel_coverage_sequence,
    resolve_gold_chunk_ids,
    spans_covered,
)
from rbi_rag.schemas import Chunk, GoldSpan, RetrievedChunk

TEXT = "Every bank shall maintain a minimum ratio of nine per cent on an ongoing basis."


def chunk(chunk_id: str, *, doc: str = "d1", start: int = 3, end: int = 3, text: str = TEXT):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc,
        text=text,
        page_start=start,
        page_end=end,
        url="https://example.test/x.PDF",
        strategy="fixed_tokens",
        target_tokens=512,
    )


def hit(chunk_id: str, rank: int, **kwargs: object) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk(chunk_id, **kwargs), score=1.0 / rank, rank=rank)  # type: ignore[arg-type]


def span(**kwargs: object) -> GoldSpan:
    base: dict[str, object] = {"doc_id": "d1", "page_start": 3, "page_end": 3}
    return GoldSpan(**{**base, **kwargs})  # type: ignore[arg-type]


class TestChunkCoversSpan:
    def test_same_document_and_page(self) -> None:
        assert chunk_covers_span(chunk("c"), span())

    def test_a_different_document_never_covers(self) -> None:
        assert not chunk_covers_span(chunk("c", doc="other"), span())

    def test_pages_must_overlap(self) -> None:
        assert not chunk_covers_span(chunk("c", start=7, end=7), span())

    def test_partial_page_overlap_counts(self) -> None:
        assert chunk_covers_span(chunk("c", start=2, end=4), span(page_start=4, page_end=6))

    def test_a_quote_must_be_present(self) -> None:
        """Page overlap alone is generous; the quote is what makes it precise."""
        assert chunk_covers_span(chunk("c"), span(quote="nine per cent"))
        assert not chunk_covers_span(chunk("c"), span(quote="twelve per cent"))

    def test_quote_matching_tolerates_line_wrapping(self) -> None:
        wrapped = chunk("c", text="Every bank shall maintain\na minimum ratio of nine per cent.")
        assert chunk_covers_span(wrapped, span(quote="maintain a minimum ratio"))


class TestRecallAtK:
    def test_all_spans_found(self) -> None:
        spans = [span(), span(page_start=4, page_end=4)]
        retrieved = [hit("a", 1), hit("b", 2, start=4, end=4)]
        assert recall_at_k(retrieved, spans, 5) == pytest.approx(1.0)

    def test_half_the_spans_found(self) -> None:
        spans = [span(), span(page_start=9, page_end=9)]
        assert recall_at_k([hit("a", 1)], spans, 5) == pytest.approx(0.5)

    def test_cut_off_at_k(self) -> None:
        spans = [span(page_start=4, page_end=4)]
        retrieved = [hit("a", 1), hit("b", 2, start=4, end=4)]
        assert recall_at_k(retrieved, spans, 1) == pytest.approx(0.0)
        assert recall_at_k(retrieved, spans, 2) == pytest.approx(1.0)

    def test_an_unanswerable_question_is_not_scored_zero(self) -> None:
        """Scoring it 0.0 would punish the system for correctly having nothing."""
        assert recall_at_k([hit("a", 1)], [], 5) is None

    def test_nothing_retrieved(self) -> None:
        assert recall_at_k([], [span()], 5) == pytest.approx(0.0)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            recall_at_k([], [span()], 0)


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank([hit("a", 1)], [span()]) == pytest.approx(1.0)

    def test_third_position(self) -> None:
        retrieved = [hit("a", 1, start=9, end=9), hit("b", 2, start=9, end=9), hit("c", 3)]
        assert reciprocal_rank(retrieved, [span()]) == pytest.approx(1 / 3)

    def test_never_found(self) -> None:
        assert reciprocal_rank([hit("a", 1, start=9, end=9)], [span()]) == pytest.approx(0.0)

    def test_unanswerable(self) -> None:
        assert reciprocal_rank([hit("a", 1)], []) is None


class TestNdcg:
    def test_perfect_ranking_scores_one(self) -> None:
        spans = [span(), span(page_start=4, page_end=4)]
        retrieved = [hit("a", 1), hit("b", 2, start=4, end=4)]
        assert ndcg_at_k(retrieved, spans, 5) == pytest.approx(1.0)

    def test_matches_the_formula(self) -> None:
        # One span, found at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
        retrieved = [hit("a", 1, start=9, end=9), hit("b", 2)]
        assert ndcg_at_k(retrieved, [span()], 5) == pytest.approx(1 / math.log2(3))

    def test_repeating_evidence_earns_nothing(self) -> None:
        """Otherwise small chunks score by returning the same span repeatedly."""
        retrieved = [hit("a", 1), hit("b", 2), hit("c", 3)]
        assert ndcg_at_k(retrieved, [span()], 5) == pytest.approx(1.0)

    def test_unanswerable(self) -> None:
        assert ndcg_at_k([hit("a", 1)], [], 5) is None


class TestChunkSizeIndependence:
    """The property the whole gold-span design exists to provide."""

    def test_the_same_evidence_scores_the_same_at_any_chunk_size(self) -> None:
        gold = [span(page_start=3, page_end=4, quote="nine per cent")]

        # Small chunks: the evidence is spread over three of them.
        small = [
            hit("s1", 1, start=3, end=3),
            hit("s2", 2, start=4, end=4, text="unrelated text about liquidity"),
            hit("s3", 3, start=4, end=4, text="more unrelated text"),
        ]
        # Large chunks: one chunk holds the lot.
        large = [hit("L1", 1, start=3, end=4)]

        assert recall_at_k(small, gold, 5) == recall_at_k(large, gold, 5) == pytest.approx(1.0)
        assert reciprocal_rank(small, gold) == reciprocal_rank(large, gold)

    def test_counting_chunks_would_have_differed(self) -> None:
        """Shows why span counting was necessary: the chunk counts disagree."""
        gold = [span(page_start=3, page_end=4, quote="nine per cent")]
        small = [chunk("s1", start=3, end=3), chunk("s2", start=4, end=4)]
        large = [chunk("L1", start=3, end=4)]
        assert len(resolve_gold_chunk_ids(small, gold)) != len(resolve_gold_chunk_ids(large, gold))


class TestCoverageHelpers:
    def test_spans_covered_returns_indices(self) -> None:
        spans = [span(), span(page_start=9, page_end=9)]
        assert spans_covered(chunk("c"), spans) == {0}

    def test_novel_coverage_only_credits_the_first(self) -> None:
        novel = novel_coverage_sequence([hit("a", 1), hit("b", 2)], [span()])
        assert novel == [{0}, set()]


class TestAggregation:
    def test_mean_skips_none(self) -> None:
        assert mean_ignoring_none([1.0, None, 0.0]) == pytest.approx(0.5)

    def test_all_none(self) -> None:
        assert mean_ignoring_none([None, None]) is None


class TestCitationPrecision:
    def test_all_valid(self) -> None:
        assert citation_precision(3, 3) == pytest.approx(1.0)

    def test_some_invalid(self) -> None:
        assert citation_precision(1, 4) == pytest.approx(0.25)

    def test_no_citations_is_not_zero(self) -> None:
        """A correct abstention cites nothing and must not be scored 0."""
        assert citation_precision(0, 0) is None

    def test_impossible_counts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonsensical"):
            citation_precision(5, 2)


class TestAbstentionRates:
    def test_both_directions_are_reported(self) -> None:
        # (is_answerable, abstained)
        rates = abstention_rates([(False, True), (False, False), (True, False), (True, True)])
        assert rates["abstention_recall"] == pytest.approx(0.5)
        assert rates["false_abstention_rate"] == pytest.approx(0.5)

    def test_always_abstaining_looks_perfect_on_one_and_terrible_on_the_other(self) -> None:
        """Why one number would be misleading on its own."""
        rates = abstention_rates([(False, True), (True, True)])
        assert rates["abstention_recall"] == pytest.approx(1.0)
        assert rates["false_abstention_rate"] == pytest.approx(1.0)

    def test_missing_categories_are_none(self) -> None:
        rates = abstention_rates([(True, False)])
        assert rates["abstention_recall"] is None


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        assert cohens_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)

    def test_total_disagreement_is_negative(self) -> None:
        kappa = cohens_kappa([True, False], [False, True])
        assert kappa is not None and kappa < 0

    def test_chance_agreement_scores_near_zero(self) -> None:
        """A judge that agrees only as often as coin flips would scores ~0."""
        judge = [True, True, False, False]
        human = [True, False, True, False]
        assert cohens_kappa(judge, human) == pytest.approx(0.0)

    def test_a_constant_judge_does_not_score_well(self) -> None:
        """Most answers are faithful; always saying so must not look good."""
        judge = [True] * 9 + [True]
        human = [True] * 9 + [False]
        kappa = cohens_kappa(judge, human)
        assert kappa is not None and kappa == pytest.approx(0.0)

    def test_too_few_items(self) -> None:
        assert cohens_kappa([True], [True]) is None

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa([True], [True, False])
