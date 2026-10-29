"""Getting PDF bytes into the raw store.

RBI serves its PDFs from ``rbidocs.rbi.org.in`` behind a JavaScript bot
challenge, so a scripted client receives a challenge page rather than a file.
This project does not attempt to solve, replay or sidestep that challenge.

Instead, acquisition is a separate step: PDFs are downloaded through a real
browser into ``data/raw/``, and everything downstream reads from there. That
keeps the pipeline fully reproducible from the raw store onward, which is what
the idempotency guarantee actually rests on. See DECISIONS.md entry 0008.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from rbi_rag.schemas import ManifestEntry

_HASH_CHUNK_BYTES = 1 << 20


class AcquisitionError(RuntimeError):
    """A PDF could not be obtained."""


def filename_for(pdf_url: str) -> str:
    """The filename a browser would save this URL as."""
    name = Path(unquote(urlparse(pdf_url).path)).name
    if not name:
        raise ValueError(f"cannot derive a filename from {pdf_url!r}")
    return name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def looks_like_pdf(path: Path) -> bool:
    """True if the file starts with the PDF magic bytes.

    Worth checking: a browser that hit the bot challenge saves an HTML page
    under a ``.PDF`` name, and that file would otherwise flow silently into the
    parser and produce an empty document.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


class RawStore:
    """The directory of downloaded PDFs, looked up by filename."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _index(self) -> dict[str, Path]:
        if not self.root.exists():
            return {}
        return {path.name.lower(): path for path in self.root.iterdir() if path.is_file()}

    def path_for(self, entry: ManifestEntry) -> Path | None:
        """Locate the PDF for an entry, or None if it has not been downloaded."""
        wanted = filename_for(entry.pdf_url).lower()
        return self._index().get(wanted)

    def missing(self, entries: Iterable[ManifestEntry]) -> list[ManifestEntry]:
        index = self._index()
        return [e for e in entries if filename_for(e.pdf_url).lower() not in index]

    def unrecognised(self, entries: Sequence[ManifestEntry]) -> list[Path]:
        """Files in the raw store that no manifest entry claims."""
        expected = {filename_for(e.pdf_url).lower() for e in entries}
        return sorted(p for name, p in self._index().items() if name not in expected)


class Fetcher(Protocol):
    """Returns the PDF bytes for a manifest entry."""

    def fetch(self, entry: ManifestEntry) -> bytes: ...


class LocalFetcher:
    """Reads PDFs that were already downloaded into the raw store."""

    def __init__(self, store: RawStore) -> None:
        self.store = store

    def fetch(self, entry: ManifestEntry) -> bytes:
        path = self.store.path_for(entry)
        if path is None:
            raise AcquisitionError(
                f"{entry.doc_id}: {filename_for(entry.pdf_url)} is not in {self.store.root}. "
                f"Download it from {entry.pdf_url} and save it there."
            )
        if not looks_like_pdf(path):
            raise AcquisitionError(
                f"{entry.doc_id}: {path.name} is not a PDF. This usually means the "
                f"download captured RBI's bot-challenge page instead of the file; "
                f"delete it and download it again from the browser."
            )
        return path.read_bytes()


class BlockedHttpFetcher:
    """Placeholder for direct HTTP download, which RBI currently prevents.

    Kept so the swap is a one-line change if RBI ever permits scripted access,
    and so the reason is discoverable from the code rather than only from docs.
    """

    def fetch(self, entry: ManifestEntry) -> bytes:
        raise AcquisitionError(
            f"Direct download of {entry.pdf_url} is not implemented. RBI's PDF host "
            f"answers scripted clients with a JavaScript bot challenge, and this "
            f"project does not circumvent it. Use browser-assisted acquisition "
            f"(see README) and the LocalFetcher."
        )
