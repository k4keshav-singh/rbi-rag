"""End-to-end ingestion against fixtures.

The Phase 1 done-criterion is that a full run is reproducible and that running
it again does nothing, so that is what most of this file checks.
"""

import urllib.request
from pathlib import Path

from rbi_rag.config import Settings
from rbi_rag.ingest import pipeline
from rbi_rag.ingest.http import PoliteClient
from rbi_rag.ingest.manifest import load_manifest
from rbi_rag.ingest.tokenize import WhitespaceTokenCounter

LISTING_URL = "https://example.test/listing"
COUNTER = WhitespaceTokenCounter()


def make_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_dir=tmp_path / "data",
        listing_url=LISTING_URL,
        chunk_size=40,
    )


def make_client(listing_html: str) -> PoliteClient:
    """A client that serves the listing fixture and nothing else."""
    responses = {
        "https://example.test/robots.txt": (404, b""),
        LISTING_URL: (200, listing_html.encode("utf-8")),
    }

    def opener(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        return responses.get(request.full_url, (404, b""))  # type: ignore[return-value]

    return PoliteClient(
        user_agent="rbi-rag/test",
        delay_seconds=2.0,
        opener=opener,
        sleeper=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )


def place_pdfs(settings: Settings, sample_pdf_bytes: bytes, names: tuple[str, ...]) -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (settings.raw_dir / name).write_bytes(sample_pdf_bytes)


ALL_PDFS = ("61MD.PDF", "29MD.PDF", "400MD.PDF")


class TestHarvest:
    def test_writes_the_manifest(self, tmp_path: Path, listing_html: str) -> None:
        settings = make_settings(tmp_path)
        report = pipeline.run_harvest(settings, make_client(listing_html))

        assert report.entries == 3
        assert report.added == 3
        assert report.rows_without_pdf == 1
        assert set(load_manifest(settings.manifest_path)) == {"11322", "10476", "13443"}

    def test_records_why_robots_was_or_was_not_honoured(
        self, tmp_path: Path, listing_html: str
    ) -> None:
        report = pipeline.run_harvest(make_settings(tmp_path), make_client(listing_html))
        assert "404" in report.robots_reason

    def test_harvesting_twice_changes_nothing(self, tmp_path: Path, listing_html: str) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        before = settings.manifest_path.read_bytes()

        second = pipeline.run_harvest(settings, make_client(listing_html))

        assert (second.added, second.updated) == (0, 0)
        assert settings.manifest_path.read_bytes() == before


class TestPdfUrlList:
    def test_lists_everything_before_acquisition(self, tmp_path: Path, listing_html: str) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        path, count = pipeline.write_pdf_url_list(settings)
        assert count == 3
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3

    def test_shrinks_as_pdfs_arrive(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ("61MD.PDF",))
        _, count = pipeline.write_pdf_url_list(settings)
        assert count == 2


class TestAcquisition:
    def test_hashes_what_is_present(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)

        report = pipeline.refresh_acquisition(settings)

        assert (report.acquired, report.missing) == (3, 0)
        entries = load_manifest(settings.manifest_path)
        assert all(e.is_acquired for e in entries.values())
        assert all(e.size_bytes == len(sample_pdf_bytes) for e in entries.values())

    def test_reports_what_is_still_missing(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ("61MD.PDF",))

        report = pipeline.refresh_acquisition(settings)
        assert (report.acquired, report.missing) == (1, 2)

    def test_flags_a_saved_challenge_page(self, tmp_path: Path, listing_html: str) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        (settings.raw_dir / "61MD.PDF").write_bytes(b"<!DOCTYPE html>")

        report = pipeline.refresh_acquisition(settings)
        assert report.not_a_pdf == ["61MD.PDF"]
        assert report.acquired == 0


class TestProcess:
    def _prepared(self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes) -> Settings:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)
        pipeline.refresh_acquisition(settings)
        return settings

    def test_writes_chunks(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = self._prepared(tmp_path, listing_html, sample_pdf_bytes)
        report = pipeline.run_process(settings, COUNTER)

        assert report.processed == 3
        assert report.failed == []
        assert report.chunks_written > 0
        assert report.output_path is not None
        assert report.output_path.exists()

    def test_enriches_the_manifest_from_the_pdf(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        """Circular number and department come off page 1, not the listing."""
        settings = self._prepared(tmp_path, listing_html, sample_pdf_bytes)
        pipeline.run_process(settings, COUNTER)

        entries = load_manifest(settings.manifest_path)
        assert entries["11322"].circular_number == "RBI/DOR/2023-24/106"
        assert entries["11322"].department == "DOR"

    def test_honours_the_limit(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = self._prepared(tmp_path, listing_html, sample_pdf_bytes)
        assert pipeline.run_process(settings, COUNTER, limit=1).processed == 1

    def test_strategy_and_size_select_the_output_file(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = self._prepared(tmp_path, listing_html, sample_pdf_bytes)
        report = pipeline.run_process(settings, COUNTER, strategy="fixed_tokens", target_tokens=64)
        assert report.output_path is not None
        assert report.output_path.name == "chunks-fixed_tokens-64.jsonl"

    def test_one_unreadable_pdf_does_not_stop_the_run(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = self._prepared(tmp_path, listing_html, sample_pdf_bytes)
        (settings.raw_dir / "29MD.PDF").write_bytes(b"%PDF- but truncated nonsense")

        report = pipeline.run_process(settings, COUNTER)

        assert len(report.failed) == 1
        assert report.processed == 2


class TestIdempotency:
    """Re-running a completed ingest must do no work and change no bytes."""

    def _run_everything(self, settings: Settings, listing_html: str) -> pipeline.ProcessReport:
        pipeline.run_harvest(settings, make_client(listing_html))
        pipeline.refresh_acquisition(settings)
        return pipeline.run_process(settings, COUNTER)

    def test_second_full_run_is_a_no_op(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)

        first = self._run_everything(settings, listing_html)
        assert first.processed == 3
        assert first.skipped_unchanged == 0

        chunks_before = first.output_path.read_bytes() if first.output_path else b""
        manifest_before = settings.manifest_path.read_bytes()

        second = self._run_everything(settings, listing_html)

        assert second.processed == 0
        assert second.skipped_unchanged == 3
        assert second.chunks_written == first.chunks_written
        assert second.output_path is not None
        assert second.output_path.read_bytes() == chunks_before
        assert settings.manifest_path.read_bytes() == manifest_before

    def test_force_reprocesses_without_changing_the_output(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)
        first = self._run_everything(settings, listing_html)
        before = first.output_path.read_bytes() if first.output_path else b""

        forced = pipeline.run_process(settings, COUNTER, force=True)

        assert forced.processed == 3
        assert forced.output_path is not None
        assert forced.output_path.read_bytes() == before

    def test_a_replaced_pdf_is_reprocessed(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)
        self._run_everything(settings, listing_html)

        # RBI reissues one Direction: same URL, different bytes.
        import pymupdf

        replacement = pymupdf.open()
        page = replacement.new_page()
        page.insert_text((72, 100), "A completely different Direction.", fontsize=11)
        (settings.raw_dir / "61MD.PDF").write_bytes(replacement.tobytes())
        replacement.close()

        second = self._run_everything(settings, listing_html)

        assert second.processed == 1
        assert second.skipped_unchanged == 2

    def test_changing_the_chunk_size_forces_rework(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)
        self._run_everything(settings, listing_html)

        resized = pipeline.run_process(settings, COUNTER, target_tokens=128)

        assert resized.processed == 3
        assert resized.skipped_unchanged == 0


class TestChunkFileStability:
    def test_rows_are_ordered_deterministically(
        self, tmp_path: Path, listing_html: str, sample_pdf_bytes: bytes
    ) -> None:
        settings = make_settings(tmp_path)
        pipeline.run_harvest(settings, make_client(listing_html))
        place_pdfs(settings, sample_pdf_bytes, ALL_PDFS)
        pipeline.refresh_acquisition(settings)

        first = pipeline.run_process(settings, COUNTER, force=True)
        assert first.output_path is not None
        contents = first.output_path.read_text(encoding="utf-8")

        second = pipeline.run_process(settings, COUNTER, force=True)
        assert second.output_path is not None
        assert second.output_path.read_text(encoding="utf-8") == contents
