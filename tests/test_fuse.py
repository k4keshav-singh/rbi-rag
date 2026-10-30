"""Reciprocal Rank Fusion.

Pure functions over id lists, which is the whole point of keeping fusion out of
SQL: these need no database and no fixtures.
"""

import pytest

from rbi_rag.retrieve.fuse import (
    DEFAULT_RRF_K,
    fuse_and_order,
    rank_positions,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    def test_single_ranking_decreases_with_position(self) -> None:
        scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_scores_match_the_formula(self) -> None:
        scores = reciprocal_rank_fusion([["a", "b"]], k=60)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)

    def test_appearing_in_both_rankings_beats_appearing_in_one(self) -> None:
        """The property that makes RRF worth using for hybrid search."""
        scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_rankings_of_different_lengths_need_no_padding(self) -> None:
        scores = reciprocal_rank_fusion([["a"], ["b", "c", "d"]], k=60)
        assert set(scores) == {"a", "b", "c", "d"}

    def test_an_empty_ranking_contributes_nothing(self) -> None:
        assert reciprocal_rank_fusion([[], ["a"]], k=60) == {"a": pytest.approx(1 / 61)}

    def test_no_rankings_gives_no_scores(self) -> None:
        assert reciprocal_rank_fusion([], k=60) == {}

    def test_a_larger_k_flattens_the_discount(self) -> None:
        small = reciprocal_rank_fusion([["a", "b"]], k=1)
        large = reciprocal_rank_fusion([["a", "b"]], k=1000)
        assert small["a"] / small["b"] > large["a"] / large["b"]

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            reciprocal_rank_fusion([["a"]], k=0)

    def test_default_k_is_the_published_value(self) -> None:
        assert DEFAULT_RRF_K == 60


class TestRankPositions:
    def test_positions_are_one_based(self) -> None:
        assert rank_positions(["a", "b"]) == {"a": 1, "b": 2}

    def test_first_occurrence_wins(self) -> None:
        assert rank_positions(["a", "b", "a"]) == {"a": 1, "b": 2}

    def test_empty(self) -> None:
        assert rank_positions([]) == {}


class TestFuseAndOrder:
    def test_orders_by_score_descending(self) -> None:
        ordered = fuse_and_order([["a", "b"], ["b", "a"]], k=60)
        assert [identifier for identifier, _ in ordered] == ["a", "b"]

    def test_ties_break_on_id_so_results_are_reproducible(self) -> None:
        """An ablation that reorders ties between runs is not reproducible."""
        first = fuse_and_order([["z", "a"], ["a", "z"]], k=60)
        second = fuse_and_order([["a", "z"], ["z", "a"]], k=60)
        assert [identifier for identifier, _ in first] == ["a", "z"]
        assert first == second
