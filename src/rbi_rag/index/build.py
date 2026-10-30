"""Build the search index from the chunk files ingestion produced.

Reads ``data/processed/chunks-<strategy>-<size>.jsonl``, embeds every chunk and
writes it into Postgres as one chunk set, then creates that set's partial HNSW
index.

Chunks longer than the embedding model's window are counted and reported. They
are still indexed -- truncated, which is what the model does -- but the count
goes into the report so a drop in recall can be traced to it rather than
mistaken for a retriever effect.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rbi_rag.config import Settings
from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import PostgresChunkStore
from rbi_rag.ingest.manifest import load_manifest
from rbi_rag.schemas import Chunk, ChunkStrategy, Document


@dataclass
class BuildReport:
    chunk_set_id: int
    documents: int
    chunks: int
    truncated: int
    index_name: str | None
    skipped_missing_documents: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"chunk set {self.chunk_set_id}: "
            f"{self.documents} document(s), {self.chunks} chunk(s) indexed"
        ]
        if self.truncated:
            lines.append(
                f"  WARNING: {self.truncated} chunk(s) exceed the embedding window and "
                f"were truncated; their tails are not represented in any vector"
            )
        if self.skipped_missing_documents:
            shown = ", ".join(self.skipped_missing_documents[:5])
            lines.append(
                f"  {len(self.skipped_missing_documents)} chunk doc_id(s) absent from the "
                f"manifest, skipped: {shown}"
            )
        if self.index_name:
            lines.append(f"  built HNSW index {self.index_name}")
        return "\n".join(lines)


def read_chunks(path: Path) -> list[Chunk]:
    """Load a chunk file written by `poe ingest process`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `uv run poe ingest process` for this "
            f"strategy and size first."
        )
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                chunks.append(Chunk.model_validate_json(stripped))
    return chunks


def current_git_sha() -> str | None:
    """The commit the index was built from, so a result can be traced to code."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def documents_for(settings: Settings, chunks: list[Chunk]) -> tuple[list[Document], list[str]]:
    """Build Document rows for the chunks, from the ingest manifest."""
    manifest = load_manifest(settings.manifest_path)
    wanted = {chunk.doc_id for chunk in chunks}
    documents: list[Document] = []
    missing: list[str] = []

    for doc_id in sorted(wanted):
        entry = manifest.get(doc_id)
        if entry is None or entry.sha256 is None:
            missing.append(doc_id)
            continue
        documents.append(
            Document(
                doc_id=entry.doc_id,
                title=entry.title,
                circular_number=entry.circular_number,
                published_on=entry.published_on,
                department=entry.department,
                url=entry.pdf_url,
                sha256=entry.sha256,
            )
        )
    return documents, missing


def build_index(
    settings: Settings,
    connection: Any,
    embedder: Embedder,
    *,
    strategy: ChunkStrategy | None = None,
    target_tokens: int | None = None,
    create_index: bool = True,
) -> BuildReport:
    """Embed and store one chunk set."""
    chosen_strategy: ChunkStrategy = strategy or settings.chunk_strategy
    chosen_size = target_tokens or settings.chunk_size

    path = settings.processed_dir / f"chunks-{chosen_strategy}-{chosen_size}.jsonl"
    chunks = read_chunks(path)

    documents, missing = documents_for(settings, chunks)
    known = {document.doc_id for document in documents}
    indexable = [chunk for chunk in chunks if chunk.doc_id in known]

    store = PostgresChunkStore(connection)
    store.upsert_documents(documents)

    chunk_set_id = store.ensure_chunk_set(
        strategy=chosen_strategy,
        target_tokens=chosen_size,
        overlap_ratio=settings.chunk_overlap_ratio,
        embedding_model=settings.embedding_model,
        embedding_dim=embedder.dimension,
        git_sha=current_git_sha(),
    )

    truncated = sum(
        1
        for chunk in indexable
        if chunk.token_count is not None and chunk.token_count > embedder.max_content_tokens
    )

    embeddings = embedder.embed_passages([chunk.text for chunk in indexable])
    store.replace_chunks(chunk_set_id, indexable, embeddings)

    index_name = store.create_hnsw_index(chunk_set_id) if create_index and indexable else None

    return BuildReport(
        chunk_set_id=chunk_set_id,
        documents=len(documents),
        chunks=len(indexable),
        truncated=truncated,
        index_name=index_name,
        skipped_missing_documents=missing,
    )


def list_chunk_sets(connection: Any) -> list[dict[str, Any]]:
    """Every chunk set currently in the database, with its size."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT s.chunk_set_id, s.strategy, s.target_tokens, s.overlap_ratio,
                   s.embedding_model, s.git_sha, count(c.chunk_id)
            FROM chunk_sets s
            LEFT JOIN chunks c USING (chunk_set_id)
            GROUP BY s.chunk_set_id
            ORDER BY s.chunk_set_id
            """
        )
        return [
            {
                "chunk_set_id": row[0],
                "strategy": row[1],
                "target_tokens": row[2],
                "overlap_ratio": row[3],
                "embedding_model": row[4],
                "git_sha": row[5],
                "chunks": row[6],
            }
            for row in cursor.fetchall()
        ]


def report_as_json(report: BuildReport) -> str:
    return json.dumps(report.__dict__, indent=2, default=str)
