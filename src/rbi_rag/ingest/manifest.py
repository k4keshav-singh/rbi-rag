"""The ingest manifest: one JSONL row per Master Direction.

The manifest is the record of what exists, where it came from, and whether its
PDF has been acquired. Re-harvesting refreshes metadata without discarding
acquisition state, which is what makes a second run a no-op.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from rbi_rag.schemas import ManifestEntry

#: Fields the listing page cannot supply. Acquisition fills the first four by
#: hashing the raw store; processing fills the last two by reading page 1 of the
#: PDF. A re-harvest must never clear them just because the listing is silent.
_ACQUISITION_FIELDS = ("pdf_filename", "sha256", "size_bytes", "acquired_at")
_DERIVED_FIELDS = (*_ACQUISITION_FIELDS, "circular_number", "department")


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Read the manifest, keyed by ``doc_id``. A missing file is an empty manifest."""
    if not path.exists():
        return {}
    entries: dict[str, ManifestEntry] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = ManifestEntry.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} is not a valid manifest row") from exc
            entries[entry.doc_id] = entry
    return entries


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    """Write the manifest atomically, sorted by ``doc_id`` for a stable diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda entry: entry.doc_id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in ordered:
            handle.write(entry.model_dump_json(exclude_none=False))
            handle.write("\n")
    os.replace(temporary, path)


def merge_harvested(
    existing: Mapping[str, ManifestEntry],
    harvested: Iterable[ManifestEntry],
) -> tuple[list[ManifestEntry], int, int]:
    """Fold freshly harvested metadata into the manifest.

    Returns the merged rows plus counts of (added, updated). Acquisition fields
    on an existing row survive, unless the PDF URL itself changed -- which means
    RBI replaced the file, so the local copy is stale and must be re-acquired.
    """
    merged = dict(existing)
    added = 0
    updated = 0

    for entry in harvested:
        current = merged.get(entry.doc_id)
        if current is None:
            merged[entry.doc_id] = entry
            added += 1
            continue

        if current.pdf_url != entry.pdf_url:
            # RBI replaced the file, so everything derived from the old bytes
            # is now about a different document. Start that row again.
            carried = entry
        else:
            carried = entry.model_copy(
                update={
                    field: getattr(entry, field) or getattr(current, field)
                    for field in _DERIVED_FIELDS
                }
            )

        if carried != current:
            merged[entry.doc_id] = carried
            updated += 1

    return sorted(merged.values(), key=lambda entry: entry.doc_id), added, updated
