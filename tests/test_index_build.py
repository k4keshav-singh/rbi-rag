"""Migration discovery and index-build preparation, without a database."""

from pathlib import Path

import pytest

from rbi_rag.config import Settings
from rbi_rag.index.build import documents_for, read_chunks
from rbi_rag.index.migrate import SCHEMA_EMBEDDING_DIM, MigrationError, discover, migrate
from rbi_rag.ingest.manifest import write_manifest
from rbi_rag.schemas import Chunk, ManifestEntry


def chunk(chunk_id: str, doc_id: str = "11322") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="Banks shall maintain a minimum ratio.",
        page_start=1,
        page_end=1,
        url="https://example.test/x.PDF",
        strategy="section_aware",
        target_tokens=512,
    )


def entry(doc_id: str, *, acquired: bool = True) -> ManifestEntry:
    return ManifestEntry(
        doc_id=doc_id,
        title=f"Direction {doc_id}",
        detail_url=f"https://www.rbi.org.in/x.aspx?id={doc_id}",
        pdf_url=f"https://rbidocs.rbi.org.in/{doc_id}.PDF",
        circular_number="RBI/DOR/2023-24/106",
        department="DOR",
        sha256=("a" * 64) if acquired else None,
    )


class TestDiscoverMigrations:
    def test_finds_the_initial_migration(self) -> None:
        versions = [m.version for m in discover()]
        assert "001_initial" in versions

    def test_order_is_by_filename(self) -> None:
        versions = [m.version for m in discover()]
        assert versions == sorted(versions)

    def test_each_migration_has_sql(self) -> None:
        assert all(m.sql.strip() for m in discover())

    def test_the_schema_creates_the_vector_extension(self) -> None:
        sql = next(m.sql for m in discover() if m.version == "001_initial")
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql

    def test_an_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert discover(tmp_path) == []


class TestEmbeddingDimGuard:
    def test_a_mismatched_dimension_is_refused_before_any_sql_runs(self) -> None:
        """Otherwise this surfaces much later as an opaque insert error."""
        with pytest.raises(MigrationError, match="needs a new migration"):
            migrate(None, embedding_dim=SCHEMA_EMBEDDING_DIM + 1)  # type: ignore[arg-type]

    def test_the_schema_dimension_matches_bge_small(self) -> None:
        assert SCHEMA_EMBEDDING_DIM == 384


class TestReadChunks:
    def test_reads_a_chunk_file(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            chunk("a").model_dump_json() + "\n" + chunk("b").model_dump_json() + "\n",
            encoding="utf-8",
        )
        assert [c.chunk_id for c in read_chunks(path)] == ["a", "b"]

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.jsonl"
        path.write_text(chunk("a").model_dump_json() + "\n\n", encoding="utf-8")
        assert len(read_chunks(path)) == 1

    def test_a_missing_file_says_what_to_run(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="poe ingest process"):
            read_chunks(tmp_path / "absent.jsonl")


class TestDocumentsFor:
    def _settings(self, tmp_path: Path) -> Settings:
        return Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]

    def test_builds_documents_from_the_manifest(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        write_manifest(settings.manifest_path, [entry("11322")])

        documents, missing = documents_for(settings, [chunk("a")])

        assert [d.doc_id for d in documents] == ["11322"]
        assert documents[0].circular_number == "RBI/DOR/2023-24/106"
        assert missing == []

    def test_a_chunk_whose_document_is_unacquired_is_reported_not_indexed(
        self, tmp_path: Path
    ) -> None:
        """Without a sha256 there is nothing to tie the chunk back to."""
        settings = self._settings(tmp_path)
        write_manifest(settings.manifest_path, [entry("11322", acquired=False)])

        documents, missing = documents_for(settings, [chunk("a")])

        assert documents == []
        assert missing == ["11322"]

    def test_a_chunk_with_no_manifest_row_at_all_is_reported(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        write_manifest(settings.manifest_path, [entry("11322")])

        _, missing = documents_for(settings, [chunk("a", doc_id="99999")])

        assert missing == ["99999"]

    def test_documents_are_deduplicated_across_chunks(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        write_manifest(settings.manifest_path, [entry("11322")])

        documents, _ = documents_for(settings, [chunk("a"), chunk("b"), chunk("c")])

        assert len(documents) == 1
