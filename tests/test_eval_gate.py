"""The CI regression gate and its snapshot.

The gate's job is to go red when retrieval changes. Most of these tests are
about the ways it could silently stay green instead.
"""

from pathlib import Path

import numpy as np
import pytest

from rbi_rag.eval.gate import arms_for, baseline_from, compare
from rbi_rag.eval.snapshot import (
    BASELINE,
    PrecomputedEmbedder,
    SnapshotError,
    SnapshotManifest,
    load_snapshot,
    write_snapshot,
)
from rbi_rag.schemas import Chunk, EvalItem, GoldSpan

ARM = "section_aware-512-hybrid-norerank"


def base(**metrics: float) -> dict[str, dict[str, float]]:
    return {ARM: dict(metrics)}


def now(**metrics: float | None) -> dict[str, dict[str, float | None]]:
    return {ARM: dict(metrics)}


class TestCompare:
    def test_identical_results_pass(self) -> None:
        report = compare(base(**{"recall@5": 0.8}), now(**{"recall@5": 0.8}), metrics=["recall@5"])
        assert report.passed
        assert "nothing regressed" in report.describe()

    def test_a_small_drop_is_tolerated(self) -> None:
        report = compare(
            base(**{"recall@5": 0.80}),
            now(**{"recall@5": 0.77}),
            metrics=["recall@5"],
            max_drop=0.05,
        )
        assert report.passed

    def test_a_drop_past_the_threshold_fails(self) -> None:
        report = compare(
            base(**{"recall@5": 0.80}),
            now(**{"recall@5": 0.70}),
            metrics=["recall@5"],
            max_drop=0.05,
        )
        assert not report.passed
        assert len(report.failures) == 1
        assert "FAILED" in report.describe()

    def test_the_threshold_is_configurable(self) -> None:
        current = now(**{"recall@5": 0.70})
        assert compare(
            base(**{"recall@5": 0.80}), current, metrics=["recall@5"], max_drop=0.2
        ).passed
        assert not compare(
            base(**{"recall@5": 0.80}), current, metrics=["recall@5"], max_drop=0.01
        ).passed

    def test_an_improvement_passes_and_is_flagged(self) -> None:
        report = compare(
            base(**{"recall@5": 0.70}), now(**{"recall@5": 0.90}), metrics=["recall@5"]
        )
        assert report.passed
        assert report.improvements
        assert "re-recording the baseline" in report.describe()

    def test_an_unmeasurable_metric_fails_rather_than_passing(self) -> None:
        """The gate cannot confirm nothing regressed, so it must not say so."""
        report = compare(base(**{"recall@5": 0.8}), now(**{"recall@5": None}), metrics=["recall@5"])
        assert not report.passed
        assert "not measurable" in report.describe()

    def test_an_arm_that_stopped_being_measured_fails(self) -> None:
        """Silently dropping an arm is a regression in the gate itself."""
        report = compare(base(**{"recall@5": 0.8}), {}, metrics=["recall@5"])
        assert not report.passed
        assert report.missing_arms == [ARM]

    def test_a_new_arm_is_noted_but_not_gated(self) -> None:
        report = compare({}, now(**{"recall@5": 0.8}), metrics=["recall@5"])
        assert report.passed
        assert report.unexpected_arms == [ARM]
        assert "not gated" in report.describe()

    def test_every_configured_metric_is_checked(self) -> None:
        report = compare(
            base(**{"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0}),
            now(**{"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0}),
            metrics=["recall@1", "recall@5", "mrr"],
        )
        assert len(report.findings) == 3

    def test_an_order_only_regression_is_caught(self) -> None:
        """The case that matters: recall@k is blind when k exceeds the corpus.

        Inverting the dense retriever on the fixture snapshot left recall@5 at
        1.0000 while MRR fell to 0.25. Gating on recall@5 alone would have
        passed a completely broken retriever.
        """
        report = compare(
            base(**{"recall@5": 1.0, "mrr": 1.0}),
            now(**{"recall@5": 1.0, "mrr": 0.25}),
            metrics=["recall@5", "mrr"],
        )
        assert not report.passed
        assert [f.metric for f in report.failures] == ["mrr"]

    def test_a_metric_absent_from_the_baseline_is_skipped(self) -> None:
        report = compare(
            base(**{"recall@5": 0.8}), now(**{"mrr": 0.1}), metrics=["recall@5", "mrr"]
        )
        assert len(report.findings) == 1

    def test_a_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            compare({}, {}, max_drop=-0.1)


class TestBaselineFrom:
    def test_keeps_measured_values(self) -> None:
        assert baseline_from(now(**{"recall@5": 0.8})) == {ARM: {"recall@5": 0.8}}

    def test_drops_unmeasurable_metrics_rather_than_recording_zero(self) -> None:
        """A baseline of 0.0 can never be regressed against; the gate would die."""
        assert baseline_from(now(**{"recall@5": None, "mrr": 0.5})) == {ARM: {"mrr": 0.5}}

    def test_an_arm_with_nothing_measurable_is_omitted(self) -> None:
        assert baseline_from(now(**{"recall@5": None})) == {}


class TestArmsFor:
    def test_names_every_gated_retriever(self) -> None:
        arms = arms_for("section_aware", 512)
        assert arms == [
            "section_aware-512-dense-norerank",
            "section_aware-512-fts-norerank",
            "section_aware-512-hybrid-norerank",
        ]

    def test_reranking_is_excluded(self) -> None:
        """The cross-encoder is a model download; the gate stays hermetic."""
        assert all("norerank" in arm for arm in arms_for("fixed_tokens", 256))


def chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        text="Banks shall maintain nine per cent.",
        page_start=3,
        page_end=3,
        circular_number="RBI/DOR/2023-24/106",
        url="https://example.test/x.PDF",
        strategy="section_aware",
        target_tokens=512,
    )


def question(item_id: str = "g0001", text: str = "what ratio?") -> EvalItem:
    return EvalItem(
        id=item_id,
        question=text,
        question_type="factual",
        gold_spans=[GoldSpan(doc_id="d1", page_start=3, page_end=3)],
    )


def manifest(**overrides: object) -> SnapshotManifest:
    base_manifest: dict[str, object] = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 4,
        "strategy": "section_aware",
        "target_tokens": 512,
        "chunk_count": 1,
        "question_count": 1,
        "created": "2026-10-30T00:00:00+00:00",
    }
    return SnapshotManifest(**{**base_manifest, **overrides})  # type: ignore[arg-type]


def build(tmp_path: Path, **overrides: object) -> Path:
    directory = tmp_path / "snapshot"
    write_snapshot(
        directory,
        manifest=manifest(**overrides),
        chunks=[chunk()],
        embeddings=np.ones((1, 4), dtype=np.float32),
        questions=[question()],
        query_vectors=np.ones((1, 4), dtype=np.float32),
        baseline={ARM: {"recall@5": 1.0}},
    )
    return directory


class TestSnapshotRoundTrip:
    def test_everything_survives(self, tmp_path: Path) -> None:
        loaded = load_snapshot(build(tmp_path))
        assert len(loaded.chunks) == 1
        assert len(loaded.questions) == 1
        assert loaded.embeddings.shape == (1, 4)
        assert loaded.baseline == {ARM: {"recall@5": 1.0}}

    def test_documents_are_derived_for_the_foreign_key(self, tmp_path: Path) -> None:
        documents = load_snapshot(build(tmp_path)).documents()
        assert [d.doc_id for d in documents] == ["d1"]

    def test_the_chunk_set_name_cannot_collide_with_a_real_index(self, tmp_path: Path) -> None:
        loaded = load_snapshot(build(tmp_path))
        assert loaded.manifest.chunk_set_model.startswith("snapshot/")

    def test_a_missing_snapshot_says_how_to_build_one(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="poe eval snapshot"):
            load_snapshot(tmp_path / "absent")

    def test_a_baseline_is_optional(self, tmp_path: Path) -> None:
        directory = build(tmp_path)
        (directory / BASELINE).unlink()
        assert load_snapshot(directory).baseline == {}


class TestStaleness:
    def test_a_different_embedding_model_is_refused(self, tmp_path: Path) -> None:
        """Committed vectors from another model would silently score the wrong thing."""
        directory = build(tmp_path)
        with pytest.raises(SnapshotError, match="different model"):
            load_snapshot(directory, expected_model="some/other-model")

    def test_a_different_dimension_is_refused(self, tmp_path: Path) -> None:
        directory = build(tmp_path)
        with pytest.raises(SnapshotError, match="dimensional"):
            load_snapshot(directory, expected_dim=768)

    def test_matching_settings_load_fine(self, tmp_path: Path) -> None:
        directory = build(tmp_path)
        assert load_snapshot(directory, expected_model="BAAI/bge-small-en-v1.5", expected_dim=4)

    def test_mismatched_counts_are_refused_at_write_time(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="chunks but"):
            write_snapshot(
                tmp_path / "s",
                manifest=manifest(),
                chunks=[chunk(), chunk("c2")],
                embeddings=np.ones((1, 4), dtype=np.float32),
                questions=[question()],
                query_vectors=np.ones((1, 4), dtype=np.float32),
                baseline={},
            )


class TestPrecomputedEmbedder:
    def _embedder(self) -> PrecomputedEmbedder:
        return PrecomputedEmbedder(
            ["what ratio?", "when to report?"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "BAAI/bge-small-en-v1.5",
        )

    def test_replays_the_committed_vector(self) -> None:
        vector = self._embedder().embed_query("what ratio?")
        assert vector.shape == (1, 2)
        assert vector[0][0] == pytest.approx(1.0)

    def test_dimension_comes_from_the_vectors(self) -> None:
        assert self._embedder().dimension == 2

    def test_an_unknown_question_is_an_error_not_a_zero_vector(self) -> None:
        """A silent zero vector would look like a retrieval regression."""
        with pytest.raises(SnapshotError, match="no committed query vector"):
            self._embedder().embed_query("a question nobody committed")

    def test_it_refuses_to_embed_passages(self) -> None:
        with pytest.raises(SnapshotError, match="cannot embed passages"):
            self._embedder().embed_passages(["text"])

    def test_mismatched_inputs_are_refused(self) -> None:
        with pytest.raises(SnapshotError, match="query vector"):
            PrecomputedEmbedder(["a", "b"], np.ones((1, 2), dtype=np.float32), "m")
