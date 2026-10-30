"""The committed index snapshot the CI gate runs against.

**This snapshot is a canary, not a measurement.** Its numbers say "retrieval
behaves as it did when the baseline was recorded", not "retrieval is good".
Quality is measured by `poe eval` against the real corpus and a human-verified
question set; nothing from here belongs in README or results/.

Query vectors are committed alongside the chunk vectors so CI needs no model
download and no network: `PrecomputedEmbedder` satisfies the Embedder protocol
by looking the question up, which lets the entire retrieval stack run
unchanged. The cost is that the gate no longer exercises the embedder itself,
so a regression inside OnnxEmbedder would not trip it -- `poe test-model`
covers that separately. The manifest records the model and dimension, and
loading fails loudly if Settings has since moved, because silently scoring an
old model's vectors would make the gate worse than useless.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rbi_rag.index.embed import FloatArray
from rbi_rag.schemas import Chunk, Document, EvalItem

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"
EMBEDDINGS = "embeddings.npy"
QUESTIONS = "questions.jsonl"
QUERY_VECTORS = "query_vectors.npy"
BASELINE = "baseline.json"

#: Keeps the snapshot's chunk set from colliding with a real index.
SNAPSHOT_MODEL_PREFIX = "snapshot/"


class SnapshotError(RuntimeError):
    """The snapshot is missing, malformed, or no longer matches the settings."""


@dataclass(frozen=True)
class SnapshotManifest:
    """What the snapshot is, and what it was built from."""

    embedding_model: str
    embedding_dim: int
    strategy: str
    target_tokens: int
    chunk_count: int
    question_count: int
    created: str
    git_sha: str | None = None

    @property
    def chunk_set_model(self) -> str:
        """Model name used for the snapshot's chunk set row."""
        return f"{SNAPSHOT_MODEL_PREFIX}{self.embedding_model}"


@dataclass(frozen=True)
class Snapshot:
    """A loaded snapshot: chunks, their vectors, questions, and query vectors."""

    manifest: SnapshotManifest
    chunks: list[Chunk]
    embeddings: FloatArray
    questions: list[EvalItem]
    query_vectors: FloatArray
    baseline: dict[str, dict[str, float]]

    def documents(self) -> list[Document]:
        """Stub document rows, so the chunks' foreign key resolves.

        Derived rather than stored: the gate never reads document metadata, and
        a second file would be one more thing to keep in step.
        """
        seen: dict[str, Document] = {}
        for chunk in self.chunks:
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = Document(
                    doc_id=chunk.doc_id,
                    title=f"snapshot document {chunk.doc_id}",
                    circular_number=chunk.circular_number,
                    url=chunk.url,
                    sha256="snapshot",
                )
        return list(seen.values())


class PrecomputedEmbedder:
    """Replays committed query vectors instead of running the model.

    Satisfies the Embedder protocol, so `retrieve()` and everything above it
    behave exactly as in production. An unknown question is an error rather
    than a zero vector: a silently wrong query embedding would look like a
    retrieval regression and send someone hunting in the wrong place.
    """

    def __init__(self, questions: list[str], vectors: FloatArray, model_name: str) -> None:
        if len(questions) != len(vectors):
            raise SnapshotError(f"{len(questions)} question(s) but {len(vectors)} query vector(s)")
        self._by_question = {question: vectors[i] for i, question in enumerate(questions)}
        self._dimension = int(vectors.shape[1]) if len(vectors) else 0
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_content_tokens(self) -> int:
        # Nothing is embedded here, so no text is ever truncated.
        return 510

    def embed_passages(self, texts: object) -> FloatArray:
        raise SnapshotError(
            "PrecomputedEmbedder cannot embed passages; it only replays committed "
            "query vectors for the CI gate."
        )

    def embed_query(self, text: str) -> FloatArray:
        vector = self._by_question.get(text)
        if vector is None:
            raise SnapshotError(
                f"no committed query vector for {text[:60]!r}. Rebuild the "
                f"snapshot with `uv run poe eval snapshot` after changing questions."
            )
        return np.asarray([vector], dtype=np.float32)


# -- writing -----------------------------------------------------------------


def write_snapshot(
    directory: Path,
    *,
    manifest: SnapshotManifest,
    chunks: list[Chunk],
    embeddings: FloatArray,
    questions: list[EvalItem],
    query_vectors: FloatArray,
    baseline: dict[str, dict[str, float]],
) -> None:
    """Write every part of the snapshot, in a stable order."""
    if len(chunks) != len(embeddings):
        raise SnapshotError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    if len(questions) != len(query_vectors):
        raise SnapshotError(f"{len(questions)} questions but {len(query_vectors)} vectors")

    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST).write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_jsonl(directory / CHUNKS, [c.model_dump_json() for c in chunks])
    _write_jsonl(directory / QUESTIONS, [q.model_dump_json() for q in questions])
    np.save(directory / EMBEDDINGS, np.asarray(embeddings, dtype=np.float32))
    np.save(directory / QUERY_VECTORS, np.asarray(query_vectors, dtype=np.float32))
    (directory / BASELINE).write_text(
        json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_jsonl(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _read_jsonl(path: Path) -> list[str]:
    if not path.exists():
        raise SnapshotError(f"{path} is missing; run `uv run poe eval snapshot`")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- loading -----------------------------------------------------------------


def load_snapshot(
    directory: Path,
    *,
    expected_model: str | None = None,
    expected_dim: int | None = None,
) -> Snapshot:
    """Load the snapshot, refusing to proceed if it has gone stale."""
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise SnapshotError(
            f"no snapshot at {directory}. Build one with `uv run poe eval snapshot`."
        )
    manifest = SnapshotManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))

    if expected_model is not None and manifest.embedding_model != expected_model:
        raise SnapshotError(
            f"snapshot was built with {manifest.embedding_model!r} but EMBEDDING_MODEL "
            f"is {expected_model!r}. The committed vectors are from a different model; "
            f"rebuild the snapshot rather than comparing across models."
        )
    if expected_dim is not None and manifest.embedding_dim != expected_dim:
        raise SnapshotError(
            f"snapshot has {manifest.embedding_dim}-dimensional vectors but "
            f"EMBEDDING_DIM is {expected_dim}."
        )

    chunks = [Chunk.model_validate_json(line) for line in _read_jsonl(directory / CHUNKS)]
    questions = [EvalItem.model_validate_json(line) for line in _read_jsonl(directory / QUESTIONS)]
    embeddings = _load_array(directory / EMBEDDINGS)
    query_vectors = _load_array(directory / QUERY_VECTORS)

    if len(chunks) != len(embeddings):
        raise SnapshotError(
            f"snapshot is inconsistent: {len(chunks)} chunks, {len(embeddings)} embeddings"
        )
    if len(questions) != len(query_vectors):
        raise SnapshotError(
            f"snapshot is inconsistent: {len(questions)} questions, "
            f"{len(query_vectors)} query vectors"
        )

    baseline_path = directory / BASELINE
    baseline: dict[str, dict[str, float]] = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    )

    return Snapshot(
        manifest=manifest,
        chunks=chunks,
        embeddings=embeddings,
        questions=questions,
        query_vectors=query_vectors,
        baseline=baseline,
    )


def _load_array(path: Path) -> FloatArray:
    if not path.exists():
        raise SnapshotError(f"{path} is missing; run `uv run poe eval snapshot`")
    return np.asarray(np.load(path), dtype=np.float32)


def now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
