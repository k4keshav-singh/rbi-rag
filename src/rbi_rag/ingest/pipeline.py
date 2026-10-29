"""Ingest orchestration: harvest, acquire, process.

The three steps are separate because they fail for different reasons and resume
differently. Harvesting needs the network; acquisition needs a browser;
processing needs neither and is the only step that must be bit-for-bit
reproducible.

Every step is idempotent. Running the whole pipeline twice over an unchanged
corpus does no work the second time, which is the Phase 1 done-criterion.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rbi_rag.config import Settings
from rbi_rag.ingest import clean, harvest, parse
from rbi_rag.ingest import manifest as manifest_io
from rbi_rag.ingest.chunk import chunk_document
from rbi_rag.ingest.fetch import LocalFetcher, RawStore, sha256_file
from rbi_rag.ingest.http import PoliteClient
from rbi_rag.ingest.tokenize import TokenCounter
from rbi_rag.schemas import Chunk, ChunkStrategy, ManifestEntry

_STATE_FILENAME = "state.json"


@dataclass
class HarvestReport:
    entries: int
    added: int
    updated: int
    rows_without_pdf: int
    robots_reason: str

    def describe(self) -> str:
        return (
            f"harvested {self.entries} entries "
            f"({self.added} new, {self.updated} changed, "
            f"{self.rows_without_pdf} rows without a usable PDF link)\n"
            f"robots.txt: {self.robots_reason}"
        )


@dataclass
class AcquisitionReport:
    total: int
    acquired: int
    missing: int
    not_a_pdf: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"{self.acquired}/{self.total} PDFs present, {self.missing} still missing"]
        if self.not_a_pdf:
            lines.append(
                f"  {len(self.not_a_pdf)} file(s) are not PDFs (likely saved bot-challenge "
                f"pages): {', '.join(self.not_a_pdf[:5])}"
            )
        if self.unrecognised:
            lines.append(f"  {len(self.unrecognised)} file(s) not referenced by the manifest")
        return "\n".join(lines)


@dataclass
class ProcessReport:
    processed: int
    skipped_unchanged: int
    failed: list[str]
    chunks_written: int
    output_path: Path | None

    def describe(self) -> str:
        lines = [
            f"processed {self.processed} document(s), "
            f"skipped {self.skipped_unchanged} unchanged, "
            f"{len(self.failed)} failed"
        ]
        if self.output_path is not None:
            lines.append(f"wrote {self.chunks_written} chunks to {self.output_path}")
        for failure in self.failed[:5]:
            lines.append(f"  failed: {failure}")
        return "\n".join(lines)


# -- harvest ---------------------------------------------------------------


def run_harvest(settings: Settings, client: PoliteClient | None = None) -> HarvestReport:
    """Fetch the listing page and fold its metadata into the manifest."""
    http = client or PoliteClient(
        user_agent=settings.user_agent,
        delay_seconds=settings.download_delay_seconds,
        max_retries=settings.download_max_retries,
    )
    decision = http.robots_decision(settings.listing_url)
    body = http.get(settings.listing_url)
    html = body.decode("utf-8", errors="replace")

    entries, rows_without_pdf = harvest.parse_listing(html, settings.listing_url)
    existing = manifest_io.load_manifest(settings.manifest_path)
    merged, added, updated = manifest_io.merge_harvested(existing, entries)
    manifest_io.write_manifest(settings.manifest_path, merged)

    return HarvestReport(
        entries=len(entries),
        added=added,
        updated=updated,
        rows_without_pdf=rows_without_pdf,
        robots_reason=decision.reason,
    )


def write_pdf_url_list(settings: Settings) -> tuple[Path, int]:
    """Write the URLs of every not-yet-acquired PDF, one per line.

    This is the handover point to browser-assisted acquisition: feed the file to
    a download manager, or open the URLs in a browser, and save into data/raw/.
    """
    entries = manifest_io.load_manifest(settings.manifest_path)
    store = RawStore(settings.raw_dir)
    outstanding = store.missing(entries.values())
    destination = settings.processed_dir.parent / "pdf_urls.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{entry.pdf_url}\n" for entry in sorted(outstanding, key=lambda e: e.doc_id)),
        encoding="utf-8",
    )
    return destination, len(outstanding)


# -- acquisition -----------------------------------------------------------


def refresh_acquisition(settings: Settings) -> AcquisitionReport:
    """Record which PDFs are now present, hashing each one."""
    entries = manifest_io.load_manifest(settings.manifest_path)
    store = RawStore(settings.raw_dir)
    from rbi_rag.ingest.fetch import looks_like_pdf

    updated: list[ManifestEntry] = []
    acquired = 0
    not_a_pdf: list[str] = []

    for entry in entries.values():
        path = store.path_for(entry)
        if path is None:
            updated.append(
                entry.model_copy(
                    update={
                        "pdf_filename": None,
                        "sha256": None,
                        "size_bytes": None,
                        "acquired_at": None,
                    }
                )
            )
            continue
        if not looks_like_pdf(path):
            not_a_pdf.append(path.name)
            updated.append(entry)
            continue

        digest = sha256_file(path)
        acquired += 1
        if entry.sha256 == digest:
            updated.append(entry)
            continue
        updated.append(
            entry.model_copy(
                update={
                    "pdf_filename": path.name,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "acquired_at": datetime.now(UTC),
                }
            )
        )

    manifest_io.write_manifest(settings.manifest_path, updated)
    return AcquisitionReport(
        total=len(entries),
        acquired=acquired,
        missing=len(entries) - acquired - len(not_a_pdf),
        not_a_pdf=not_a_pdf,
        unrecognised=[p.name for p in store.unrecognised(list(entries.values()))],
    )


# -- processing ------------------------------------------------------------


def _state_path(settings: Settings) -> Path:
    return settings.processed_dir / _STATE_FILENAME


def _load_state(settings: Settings) -> dict[str, dict[str, object]]:
    path = _state_path(settings)
    if not path.exists():
        return {}
    loaded: dict[str, dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _save_state(settings: Settings, state: dict[str, dict[str, object]]) -> None:
    path = _state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def chunks_output_path(settings: Settings, strategy: ChunkStrategy, size: int) -> Path:
    return settings.processed_dir / f"chunks-{strategy}-{size}.jsonl"


def process_document(
    entry: ManifestEntry,
    pdf_bytes: bytes,
    *,
    strategy: ChunkStrategy,
    target_tokens: int,
    counter: TokenCounter,
    overlap_ratio: float,
) -> tuple[list[Chunk], parse.CircularMetadata]:
    """Parse, clean and chunk a single PDF."""
    pages = parse.extract_pages(pdf_bytes)
    cleaned = clean.clean_pages(pages)
    outline = parse.extract_outline(pdf_bytes)
    metadata = parse.extract_circular_metadata(pages[0].text if pages else "")

    chunks = chunk_document(
        doc_id=entry.doc_id,
        url=entry.pdf_url,
        pages=cleaned,
        strategy=strategy,
        target_tokens=target_tokens,
        counter=counter,
        circular_number=metadata.circular_number or entry.circular_number,
        outline=outline,
        overlap_ratio=overlap_ratio,
    )
    return chunks, metadata


def run_process(
    settings: Settings,
    counter: TokenCounter,
    *,
    strategy: ChunkStrategy | None = None,
    target_tokens: int | None = None,
    force: bool = False,
    limit: int | None = None,
) -> ProcessReport:
    """Chunk every acquired document whose output is missing or stale."""
    chosen_strategy: ChunkStrategy = strategy or settings.chunk_strategy
    chosen_size = target_tokens or settings.chunk_size

    entries = manifest_io.load_manifest(settings.manifest_path)
    store = RawStore(settings.raw_dir)
    fetcher = LocalFetcher(store)

    acquired = [e for e in entries.values() if e.is_acquired]
    acquired.sort(key=lambda e: e.doc_id)
    if limit is not None:
        acquired = acquired[:limit]

    state = {} if force else _load_state(settings)
    signature = f"{chosen_strategy}:{chosen_size}:{settings.chunk_overlap_ratio}"

    all_chunks: list[Chunk] = []
    enriched: dict[str, ManifestEntry] = dict(entries)
    processed = 0
    skipped = 0
    failed: list[str] = []
    new_state: dict[str, dict[str, object]] = dict(state)

    for entry in acquired:
        previous = state.get(entry.doc_id)
        unchanged = (
            previous is not None
            and previous.get("sha256") == entry.sha256
            and previous.get("signature") == signature
        )
        try:
            chunks, metadata = process_document(
                entry,
                fetcher.fetch(entry),
                strategy=chosen_strategy,
                target_tokens=chosen_size,
                counter=counter,
                overlap_ratio=settings.chunk_overlap_ratio,
            )
        except Exception as exc:
            failed.append(f"{entry.doc_id}: {exc}")
            continue

        all_chunks.extend(chunks)
        if unchanged:
            skipped += 1
        else:
            processed += 1
            if metadata.circular_number or metadata.department:
                enriched[entry.doc_id] = entry.model_copy(
                    update={
                        "circular_number": metadata.circular_number or entry.circular_number,
                        "department": metadata.department or entry.department,
                    }
                )
        new_state[entry.doc_id] = {
            "sha256": entry.sha256,
            "signature": signature,
            "chunks": len(chunks),
        }

    output_path: Path | None = None
    if all_chunks:
        output_path = chunks_output_path(settings, chosen_strategy, chosen_size)
        write_chunks(output_path, all_chunks)
        manifest_io.write_manifest(settings.manifest_path, enriched.values())
        _save_state(settings, new_state)

    return ProcessReport(
        processed=processed,
        skipped_unchanged=skipped,
        failed=failed,
        chunks_written=len(all_chunks),
        output_path=output_path,
    )


def write_chunks(path: Path, chunks: Sequence[Chunk]) -> None:
    """Write chunks as JSONL, ordered so the file is stable across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(chunks, key=lambda c: (c.doc_id, c.chunk_id))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in ordered:
            handle.write(chunk.model_dump_json())
            handle.write("\n")
