"""The eval set, the response cache, and results rendering.

Rule 2 of this project is that nothing automated may mark an item verified.
The first class here is the one that enforces it.
"""

from pathlib import Path

import pytest

from rbi_rag.eval.cache import ResponseCache, cache_key
from rbi_rag.eval.calibration import calibrate, interpret
from rbi_rag.eval.dataset import (
    UnverifiedItemError,
    load_eval_set,
    next_item_id,
    summarise,
    verified_only,
    write_drafted,
    write_eval_set,
)
from rbi_rag.eval.report import ArmResult, arm_rows, format_value, render_markdown, write_csv
from rbi_rag.schemas import EvalItem, GoldSpan


def item(item_id: str, *, verified: bool = False, question_type: str = "factual") -> EvalItem:
    return EvalItem(
        id=item_id,
        question=f"question {item_id}",
        question_type=question_type,  # type: ignore[arg-type]
        gold_spans=[GoldSpan(doc_id="d1", page_start=1, page_end=2)],
        verified_by_human=verified,
    )


class TestVerificationCannotBeAutomated:
    def test_drafting_refuses_to_write_a_verified_item(self, tmp_path: Path) -> None:
        with pytest.raises(UnverifiedItemError, match="may not be verified"):
            write_drafted(tmp_path / "q.jsonl", [item("f0001", verified=True)])

    def test_the_offending_ids_are_named(self, tmp_path: Path) -> None:
        with pytest.raises(UnverifiedItemError, match="f0002"):
            write_drafted(tmp_path / "q.jsonl", [item("f0001"), item("f0002", verified=True)])

    def test_a_redraft_never_overwrites_a_reviewed_item(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        write_eval_set(path, [item("f0001", verified=True)])

        added = write_drafted(path, [item("f0001")])

        assert added == 0
        assert load_eval_set(path)[0].verified_by_human is True

    def test_metrics_see_only_verified_items(self) -> None:
        items = [item("f0001", verified=True), item("f0002")]
        assert [i.id for i in verified_only(items)] == ["f0001"]


class TestRoundTrip:
    def test_a_missing_file_is_an_empty_set(self, tmp_path: Path) -> None:
        assert load_eval_set(tmp_path / "absent.jsonl") == []

    def test_items_survive(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        original = [item("f0001"), item("u0001", question_type="unanswerable")]
        write_eval_set(path, original)
        assert load_eval_set(path) == sorted(original, key=lambda i: i.id)

    def test_writing_twice_gives_identical_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        write_eval_set(path, [item("f0002"), item("f0001")])
        first = path.read_bytes()
        write_eval_set(path, [item("f0001"), item("f0002")])
        assert path.read_bytes() == first

    def test_a_corrupt_row_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text('{"id": "f1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":1 is not a valid eval item"):
            load_eval_set(path)

    def test_new_items_are_added(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        write_eval_set(path, [item("f0001")])
        assert write_drafted(path, [item("f0002")]) == 1
        assert len(load_eval_set(path)) == 2


class TestSummary:
    def test_counts_by_state_and_type(self) -> None:
        counts = summarise(
            [
                item("f0001", verified=True),
                item("u0001", verified=True, question_type="unanswerable"),
                item("f0002"),
            ]
        )
        assert counts["total"] == 3
        assert counts["verified"] == 2
        assert counts["unverified"] == 1
        assert counts["verified_unanswerable"] == 1

    def test_ids_do_not_collide(self) -> None:
        existing = [item("f0001"), item("f0002")]
        assert next_item_id(existing, "factual") == "f0003"

    def test_ids_are_prefixed_by_type(self) -> None:
        assert next_item_id([], "unanswerable").startswith("u")


class TestResponseCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("abc123", {"faithful": True})
        assert cache.get("abc123") == {"faithful": True}

    def test_a_miss_returns_none(self, tmp_path: Path) -> None:
        assert ResponseCache(tmp_path).get("nope") is None

    def test_disabled_cache_never_serves(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, enabled=False)
        cache.put("abc", {"x": 1})
        assert cache.get("abc") is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("abc123", {"ok": True})
        path = tmp_path / "ab" / "abc123.json"
        path.write_text("{not json", encoding="utf-8")
        assert cache.get("abc123") is None

    def test_hit_rate_is_tracked(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("k", {"v": 1})
        cache.get("k")
        cache.get("missing")
        assert "1/2" in cache.describe()


class TestCacheKey:
    def test_is_stable(self) -> None:
        assert cache_key(a=1, b="x") == cache_key(a=1, b="x")

    def test_ignores_argument_order(self) -> None:
        assert cache_key(a=1, b=2) == cache_key(b=2, a=1)

    def test_changes_with_content(self) -> None:
        """A different answer must not reuse a verdict about the old one."""
        assert cache_key(answer="nine per cent") != cache_key(answer="twelve per cent")


class TestCalibrationReport:
    def test_no_labels_says_todo(self) -> None:
        assert "TODO" in calibrate([], []).describe()

    def test_reports_kappa_and_agreement(self) -> None:
        report = calibrate([True, False, True, False], [True, False, False, False])
        assert report.n == 4
        assert report.kappa is not None
        assert "kappa=" in report.describe()

    def test_a_constant_labeller_is_flagged_as_degenerate(self) -> None:
        report = calibrate([True, True, True], [True, True, True])
        assert report.degenerate
        assert "degenerate" in report.describe()

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            calibrate([True], [True, False])

    @pytest.mark.parametrize(
        ("kappa", "label"),
        [(0.9, "almost perfect"), (0.7, "substantial"), (0.5, "moderate"), (-0.1, "poor")],
    )
    def test_interpretation_bands(self, kappa: float, label: str) -> None:
        assert interpret(kappa) == label


class TestReportRendering:
    def test_unmeasured_values_render_as_todo(self) -> None:
        """Never a blank, which reads like zero."""
        assert format_value(None) == "TODO"

    def test_floats_are_fixed_width(self) -> None:
        assert format_value(0.5) == "0.5000"

    def test_markdown_has_a_header_and_rows(self) -> None:
        rendered = render_markdown(
            "Retrieval",
            {"git_sha": "abc"},
            ["arm", "recall@5"],
            [{"arm": "x", "recall@5": 0.5}],
        )
        assert "# Retrieval" in rendered
        assert "**git_sha**: abc" in rendered
        assert "| x | 0.5000 |" in rendered

    def test_an_empty_table_says_so(self) -> None:
        rendered = render_markdown("Retrieval", {}, ["arm"], [])
        assert "nothing measurable yet" in rendered

    def test_notes_are_rendered(self) -> None:
        rendered = render_markdown("T", {}, ["a"], [{"a": 1}], notes=["ts_rank_cd is not BM25"])
        assert "ts_rank_cd is not BM25" in rendered

    def test_arm_rows_carry_config_and_metrics(self) -> None:
        result = ArmResult("section_aware", 512, "hybrid", True, 1, items=10)
        result.metrics = {"recall@5": 0.8}
        rows = arm_rows([result], ["recall@5"])
        assert rows[0]["arm"] == "section_aware-512-hybrid-rerank"
        assert rows[0]["recall@5"] == 0.8

    def test_csv_writes_none_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        write_csv(path, ["arm", "recall@5"], [{"arm": "x", "recall@5": None}])
        assert path.read_text(encoding="utf-8").splitlines()[1] == "x,"
