"""Reciprocal Rank Fusion.

Deliberately a pure function over lists of ids rather than SQL. Dense scores
are cosine similarities in [-1, 1] and keyword scores are ``ts_rank_cd`` values
on an unbounded scale, so the two cannot be combined by arithmetic without
inventing a normalisation. RRF sidesteps that by using only rank positions,
which is why it is the fusion step here.

Keeping it in Python costs one extra database round trip and buys unit tests
that need no database at all.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Cormack et al. (2009) use 60; large values flatten the rank discount.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Fuse ranked id lists into one score per id.

    Each list contributes ``1 / (k + rank)`` for every id it ranks, with rank
    counted from 1. An id absent from a list simply contributes nothing, so
    lists of different lengths need no padding.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + position)
    return scores


def rank_positions(ranking: Sequence[str]) -> dict[str, int]:
    """Map each id to its 1-based position, keeping the first occurrence."""
    positions: dict[str, int] = {}
    for position, identifier in enumerate(ranking, start=1):
        positions.setdefault(identifier, position)
    return positions


def fuse_and_order(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Fuse, then order by score descending with the id as a stable tiebreak."""
    scores = reciprocal_rank_fusion(rankings, k)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
