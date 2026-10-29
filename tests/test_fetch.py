"""The raw store and the fetchers."""

from pathlib import Path

import pytest

from rbi_rag.ingest.fetch import (
    AcquisitionError,
    BlockedHttpFetcher,
    LocalFetcher,
    RawStore,
    filename_for,
    looks_like_pdf,
    sha256_bytes,
    sha256_file,
)
from rbi_rag.schemas import ManifestEntry


def entry(doc_id: str = "1", pdf: str = "61MD.PDF") -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        title="Direction",
        detail_url="https://www.rbi.org.in/x.aspx?id=1",
        pdf_url=f"https://rbidocs.rbi.org.in/rdocs/notification/PDFs/{pdf}",
    )


class TestFilenameFor:
    def test_takes_the_basename(self) -> None:
        assert filename_for("https://rbidocs.rbi.org.in/rdocs/x/61MD.PDF") == "61MD.PDF"

    def test_decodes_percent_escapes(self) -> None:
        assert filename_for("https://example.test/a%20b.PDF") == "a b.PDF"

    def test_rejects_a_url_with_no_filename(self) -> None:
        with pytest.raises(ValueError, match="cannot derive a filename"):
            filename_for("https://example.test/")


class TestHashing:
    def test_bytes_and_file_agree(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello")
        assert sha256_file(path) == sha256_bytes(b"hello")

    def test_is_stable(self) -> None:
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")


class TestLooksLikePdf:
    def test_accepts_a_real_pdf(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        path = tmp_path / "a.pdf"
        path.write_bytes(sample_pdf_bytes)
        assert looks_like_pdf(path)

    def test_rejects_a_saved_html_page(self, tmp_path: Path) -> None:
        """A browser that hit the bot challenge saves HTML under a .PDF name."""
        path = tmp_path / "a.PDF"
        path.write_bytes(b"<!DOCTYPE html><html>challenge</html>")
        assert not looks_like_pdf(path)

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        assert not looks_like_pdf(tmp_path / "absent.pdf")


class TestRawStore:
    def test_finds_a_downloaded_file(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        (tmp_path / "61MD.PDF").write_bytes(sample_pdf_bytes)
        assert RawStore(tmp_path).path_for(entry()) is not None

    def test_matching_ignores_case(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        (tmp_path / "61md.pdf").write_bytes(sample_pdf_bytes)
        assert RawStore(tmp_path).path_for(entry()) is not None

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        assert RawStore(tmp_path).path_for(entry()) is None

    def test_a_missing_directory_is_simply_empty(self, tmp_path: Path) -> None:
        store = RawStore(tmp_path / "nothing-here")
        assert store.path_for(entry()) is None
        assert store.missing([entry()]) == [entry()]

    def test_lists_what_is_still_missing(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        (tmp_path / "61MD.PDF").write_bytes(sample_pdf_bytes)
        entries = [entry("1", "61MD.PDF"), entry("2", "29MD.PDF")]
        assert [e.doc_id for e in RawStore(tmp_path).missing(entries)] == ["2"]

    def test_lists_files_the_manifest_does_not_claim(self, tmp_path: Path) -> None:
        (tmp_path / "stray.PDF").write_bytes(b"%PDF-")
        assert [p.name for p in RawStore(tmp_path).unrecognised([entry()])] == ["stray.PDF"]


class TestLocalFetcher:
    def test_returns_the_bytes(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        (tmp_path / "61MD.PDF").write_bytes(sample_pdf_bytes)
        assert LocalFetcher(RawStore(tmp_path)).fetch(entry()) == sample_pdf_bytes

    def test_explains_how_to_obtain_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AcquisitionError, match="Download it from"):
            LocalFetcher(RawStore(tmp_path)).fetch(entry())

    def test_refuses_a_challenge_page_saved_as_a_pdf(self, tmp_path: Path) -> None:
        (tmp_path / "61MD.PDF").write_bytes(b"<!DOCTYPE html>")
        with pytest.raises(AcquisitionError, match="bot-challenge page"):
            LocalFetcher(RawStore(tmp_path)).fetch(entry())


class TestBlockedHttpFetcher:
    def test_refuses_and_says_why(self) -> None:
        with pytest.raises(AcquisitionError, match="does not circumvent"):
            BlockedHttpFetcher().fetch(entry())
