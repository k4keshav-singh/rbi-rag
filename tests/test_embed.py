"""Embedding.

The pure maths is unit-tested here. Anything that loads the real model is
marked ``model`` because it downloads from HuggingFace on first run, and no
test in the default suite may touch the network.
"""

import numpy as np
import pytest

from rbi_rag.index.embed import l2_normalise


class TestL2Normalise:
    def test_rows_become_unit_length(self) -> None:
        normalised = l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
        assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)

    def test_direction_is_preserved(self) -> None:
        normalised = l2_normalise(np.array([[3.0, 4.0]], dtype=np.float32))
        assert normalised[0][0] == pytest.approx(0.6)
        assert normalised[0][1] == pytest.approx(0.8)

    def test_a_zero_row_does_not_become_nan(self) -> None:
        """Dividing by a zero norm would poison every later comparison."""
        normalised = l2_normalise(np.array([[0.0, 0.0]], dtype=np.float32))
        assert not np.isnan(normalised).any()
        assert np.array_equal(normalised, np.array([[0.0, 0.0]], dtype=np.float32))

    def test_normalising_twice_changes_nothing(self) -> None:
        once = l2_normalise(np.array([[3.0, 4.0]], dtype=np.float32))
        assert np.allclose(l2_normalise(once), once)

    def test_output_is_float32(self) -> None:
        """Postgres gets a vector(384) of real4; float64 would be silently cast."""
        normalised = l2_normalise(np.array([[1.0, 2.0]], dtype=np.float64))  # type: ignore[arg-type]
        assert normalised.dtype == np.float32

    def test_an_empty_matrix_is_handled(self) -> None:
        assert l2_normalise(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)


@pytest.fixture(scope="module")
def embedder() -> object:
    """The real ONNX embedder. Downloads the model on first run."""
    from rbi_rag.config import get_settings
    from rbi_rag.index.embed import OnnxEmbedder

    settings = get_settings()
    return OnnxEmbedder(settings.embedding_model, onnx_file=settings.onnx_file)


@pytest.mark.model
class TestOnnxEmbedder:
    """Needs the real model. Run with `uv run poe test-model`."""

    def test_dimension_matches_the_configured_one(self, embedder: object) -> None:
        from rbi_rag.config import get_settings

        assert embedder.dimension == get_settings().embedding_dim  # type: ignore[attr-defined]

    def test_the_sequence_limit_is_read_from_the_model(self, embedder: object) -> None:
        """Measured, not assumed: this is what truncates 1024-token chunks."""
        assert embedder.max_sequence_length == 512  # type: ignore[attr-defined]
        assert embedder.max_content_tokens == 510  # type: ignore[attr-defined]

    def test_embeddings_are_unit_length(self, embedder: object) -> None:
        vectors = embedder.embed_passages(["capital adequacy", "liquidity"])  # type: ignore[attr-defined]
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_related_text_scores_higher_than_unrelated(self, embedder: object) -> None:
        vectors = embedder.embed_passages(  # type: ignore[attr-defined]
            [
                "banks shall maintain a minimum capital adequacy ratio",
                "the capital to risk weighted assets ratio must be nine per cent",
                "a recipe for lemon drizzle cake",
            ]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated

    def test_embedding_is_deterministic(self, embedder: object) -> None:
        first = embedder.embed_passages(["capital adequacy"])  # type: ignore[attr-defined]
        second = embedder.embed_passages(["capital adequacy"])  # type: ignore[attr-defined]
        assert np.allclose(first, second)

    def test_batching_does_not_change_the_result(self, embedder: object) -> None:
        """Padding differs between batch shapes; attention masking must absorb it."""
        texts = ["short text", "a considerably longer passage about capital adequacy rules"]
        together = embedder.embed_passages(texts)  # type: ignore[attr-defined]
        apart = np.vstack([embedder.embed_passages([t]) for t in texts])  # type: ignore[attr-defined]
        assert np.allclose(together, apart, atol=1e-4)

    def test_no_text_gives_no_vectors(self, embedder: object) -> None:
        assert embedder.embed_passages([]).shape[0] == 0  # type: ignore[attr-defined]
