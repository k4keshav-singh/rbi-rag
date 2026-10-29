"""The manifest: round-tripping, and what survives a re-harvest."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rbi_rag.ingest.manifest import load_manifest, merge_harvested, write_manifest
from rbi_rag.schemas import ManifestEntry


def entry(doc_id: str = "1", **overrides: object) -> ManifestEntry:
    defaults: dict[str, object] = {
        "doc_id": doc_id,
        "title": f"Direction {doc_id}",
        "detail_url": f"https://www.rbi.org.in/x.aspx?id={doc_id}",
        "pdf_url": f"https://rbidocs.rbi.org.in/{doc_id}.PDF",
    }
    return ManifestEntry(**{**defaults, **overrides})  # type: ignore[arg-type]


def acquired(doc_id: str = "1", **overrides: object) -> ManifestEntry:
    return entry(
        doc_id,
        pdf_filename=f"{doc_id}.PDF",
        sha256="a" * 64,
        size_bytes=1234,
        acquired_at=datetime(2026, 1, 1, tzinfo=UTC),
        **overrides,
    )


class TestRoundTrip:
    def test_a_missing_file_is_an_empty_manifest(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path / "nope.jsonl") == {}

    def test_entries_survive_a_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        original = [acquired("1"), entry("2")]
        write_manifest(path, original)
        assert load_manifest(path) == {e.doc_id: e for e in original}

    def test_rows_are_sorted_for_a_stable_diff(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        write_manifest(path, [entry("9"), entry("2"), entry("5")])
        ids = [line.split('"doc_id":"')[1][:1] for line in path.read_text().splitlines()]
        assert ids == ["2", "5", "9"]

    def test_writing_twice_produces_identical_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        write_manifest(path, [entry("1"), acquired("2")])
        first = path.read_bytes()
        write_manifest(path, [acquired("2"), entry("1")])
        assert path.read_bytes() == first

    def test_a_corrupt_row_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text('{"doc_id": "1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":1 is not a valid manifest row"):
            load_manifest(path)

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        write_manifest(path, [entry("1")])
        path.write_text(path.read_text() + "\n\n", encoding="utf-8")
        assert len(load_manifest(path)) == 1


class TestMergeHarvested:
    def test_new_entries_are_added(self) -> None:
        merged, added, updated = merge_harvested({}, [entry("1"), entry("2")])
        assert len(merged) == 2
        assert (added, updated) == (2, 0)

    def test_acquisition_state_survives_a_reharvest(self) -> None:
        """The listing page knows nothing about downloads; it must not clear them."""
        existing = {"1": acquired("1")}
        merged, added, updated = merge_harvested(existing, [entry("1")])
        assert merged[0].sha256 == "a" * 64
        assert merged[0].pdf_filename == "1.PDF"
        assert (added, updated) == (0, 0)

    def test_refreshed_metadata_is_taken(self) -> None:
        existing = {"1": acquired("1")}
        merged, _, updated = merge_harvested(existing, [entry("1", title="Renamed")])
        assert merged[0].title == "Renamed"
        assert merged[0].sha256 == "a" * 64
        assert updated == 1

    def test_a_changed_pdf_url_invalidates_the_local_copy(self) -> None:
        """RBI replaced the file, so the hash we hold is for the wrong document."""
        existing = {"1": acquired("1")}
        replacement = entry("1", pdf_url="https://rbidocs.rbi.org.in/1-v2.PDF")
        merged, _, updated = merge_harvested(existing, [replacement])
        assert merged[0].sha256 is None
        assert merged[0].pdf_filename is None
        assert updated == 1

    def test_an_unchanged_reharvest_reports_no_work(self) -> None:
        existing = {"1": entry("1"), "2": entry("2")}
        _, added, updated = merge_harvested(existing, [entry("1"), entry("2")])
        assert (added, updated) == (0, 0)

    def test_entries_absent_from_the_listing_are_retained(self) -> None:
        """A Direction dropping off the page should not delete what we already have."""
        existing = {"1": acquired("1")}
        merged, _, _ = merge_harvested(existing, [entry("2")])
        assert {e.doc_id for e in merged} == {"1", "2"}
