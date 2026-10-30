"""ONNX versus sentence-transformers.

The ONNX backend reimplements CLS pooling and L2 normalisation rather than
inheriting them from sentence-transformers. That is the one place in this
project where a silent, plausible-looking error could degrade every retrieval
number without anything failing, so it is checked against the reference
implementation rather than reasoned about.

Needs torch::

    uv sync --extra ml
    uv run poe test-parity

Not in CI: installing torch on every push is exactly the cost the ONNX backend
exists to avoid. See DECISIONS.md entry 0014.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.parity

SENTENCES = [
    "Banks shall maintain a minimum capital adequacy ratio of nine per cent.",
    "Tier 1 capital shall not be less than seven per cent of risk weighted assets.",
    "Every bank shall report its capital position within fifteen days.",
    "a short one",
    "A considerably longer passage that exercises padding behaviour across a batch, "
    "repeated and extended so that the two backends must agree on how attention "
    "masking interacts with sequences of markedly different lengths.",
]


@pytest.fixture(scope="module")
def backends() -> tuple[object, object]:
    from rbi_rag.config import get_settings
    from rbi_rag.index.embed import OnnxEmbedder, SentenceTransformerEmbedder

    settings = get_settings()
    pytest.importorskip("sentence_transformers", reason="needs: uv sync --extra ml")
    onnx = OnnxEmbedder(settings.embedding_model, onnx_file=settings.onnx_file)
    reference = SentenceTransformerEmbedder(settings.embedding_model)
    return onnx, reference


def test_dimensions_agree(backends: tuple[object, object]) -> None:
    onnx, reference = backends
    assert onnx.dimension == reference.dimension  # type: ignore[attr-defined]


def test_sequence_limits_agree(backends: tuple[object, object]) -> None:
    onnx, reference = backends
    assert onnx.max_content_tokens == reference.max_content_tokens  # type: ignore[attr-defined]


def test_vectors_match_elementwise(backends: tuple[object, object]) -> None:
    onnx, reference = backends
    ours = onnx.embed_passages(SENTENCES)  # type: ignore[attr-defined]
    theirs = reference.embed_passages(SENTENCES)  # type: ignore[attr-defined]
    assert np.allclose(ours, theirs, atol=1e-4), (
        f"max elementwise difference {np.abs(ours - theirs).max():.2e}"
    )


def test_cosine_similarity_is_effectively_one(backends: tuple[object, object]) -> None:
    """The measure that actually matters: retrieval compares by cosine."""
    onnx, reference = backends
    ours = onnx.embed_passages(SENTENCES)  # type: ignore[attr-defined]
    theirs = reference.embed_passages(SENTENCES)  # type: ignore[attr-defined]
    similarities = np.sum(ours * theirs, axis=1)
    assert similarities.min() > 0.9999, f"worst cosine similarity {similarities.min():.6f}"


def test_relative_ranking_is_identical(backends: tuple[object, object]) -> None:
    """Even a tiny numeric drift must not reorder results."""
    onnx, reference = backends
    query = "what is the minimum capital adequacy ratio"
    ours = onnx.embed_passages(SENTENCES) @ onnx.embed_query(query)[0]  # type: ignore[attr-defined]
    theirs = reference.embed_passages(SENTENCES) @ reference.embed_query(query)[0]  # type: ignore[attr-defined]
    assert list(np.argsort(-ours)) == list(np.argsort(-theirs))
