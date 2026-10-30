"""Retrieval and generation metrics.

Every retrieval metric is computed over **gold spans**, not over chunks. That
choice is forced by the ablation: the number of chunks covering a fixed piece
of evidence falls as chunk size rises, so a chunk-counting recall would make
1024-token chunks look better purely because there are fewer of them to find.
Counting spans asks the question that actually matters -- was the evidence
surfaced in the top k -- and gives a number comparable across configurations.

All functions are pure, so they are tested against hand-computed values with no
database, no model and no network.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rbi_rag.eval.relevance import coverage_sequence, novel_coverage_sequence
from rbi_rag.schemas import GoldSpan, RetrievedChunk


def recall_at_k(
    retrieved: Sequence[RetrievedChunk],
    spans: Sequence[GoldSpan],
    k: int,
) -> float | None:
    """Fraction of gold spans covered by at least one chunk in the top k.

    Returns None when there are no gold spans -- an unanswerable question has
    nothing to recall, and scoring it 0.0 would drag the average down for
    answering correctly.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not spans:
        return None
    found: set[int] = set()
    for covered in coverage_sequence(retrieved[:k], spans):
        found |= covered
    return len(found) / len(spans)


def reciprocal_rank(
    retrieved: Sequence[RetrievedChunk],
    spans: Sequence[GoldSpan],
) -> float | None:
    """1 / rank of the first chunk covering any gold span; 0.0 if none does."""
    if not spans:
        return None
    for position, covered in enumerate(coverage_sequence(retrieved, spans), start=1):
        if covered:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[RetrievedChunk],
    spans: Sequence[GoldSpan],
    k: int,
) -> float | None:
    """Normalised discounted cumulative gain over gold spans.

    Gain is 1 where a chunk is the first to cover a span and 0 otherwise, so
    re-finding evidence already surfaced earns nothing. The ideal ranking puts
    one newly-covered span at each of the first ``min(k, len(spans))`` ranks.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not spans:
        return None

    gains = [1.0 if fresh else 0.0 for fresh in novel_coverage_sequence(retrieved[:k], spans)]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))

    ideal_hits = min(k, len(spans))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean_ignoring_none(values: Sequence[float | None]) -> float | None:
    """Average the values that exist. None when nothing is measurable."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


# -- generation --------------------------------------------------------------


def citation_precision(valid: int, claimed: int) -> float | None:
    """Share of the citations an answer made that survived validation.

    None when the answer cited nothing: there is no precision to report, and
    counting it as 0.0 would punish a correct abstention.
    """
    if claimed < 0 or valid < 0 or valid > claimed:
        raise ValueError(f"nonsensical citation counts: {valid} valid of {claimed} claimed")
    if claimed == 0:
        return None
    return valid / claimed


def abstention_rates(
    outcomes: Sequence[tuple[bool, bool]],
) -> dict[str, float | None]:
    """Abstention behaviour, split by whether the question was answerable.

    Takes (is_answerable, abstained) pairs. Reported as two numbers because one
    would hide the trade: a system that abstains on everything scores perfectly
    on unanswerable questions and is useless.
    """
    unanswerable = [abstained for answerable, abstained in outcomes if not answerable]
    answerable = [abstained for answerable, abstained in outcomes if answerable]

    return {
        # Share of unanswerable questions correctly declined.
        "abstention_recall": (sum(unanswerable) / len(unanswerable) if unanswerable else None),
        # Share of answerable questions wrongly declined.
        "false_abstention_rate": (sum(answerable) / len(answerable) if answerable else None),
        "unanswerable_count": float(len(unanswerable)),
        "answerable_count": float(len(answerable)),
    }


def cohens_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float | None:
    """Agreement between two binary labellers, corrected for chance.

    Used to calibrate the LLM judge against the owner's manual labels. Reported
    beside the faithfulness score, because a faithfulness number from an
    uncalibrated judge says more about the judge than about the system.

    Returns None for fewer than two items. Where both labellers are perfectly
    constant and identical, chance agreement is 1.0 and kappa is undefined;
    that case returns 1.0, which is the conventional reading of "they agreed on
    everything" and is flagged in the report as degenerate.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    total = len(labels_a)
    if total < 2:
        return None

    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / total

    a_true = sum(labels_a) / total
    b_true = sum(labels_b) / total
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)

    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)
