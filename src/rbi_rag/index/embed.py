"""Embedding with BAAI/bge-small-en-v1.5.

The default backend is onnxruntime rather than sentence-transformers, which
keeps torch out of the dependency tree entirely (~130 MB against ~2.5 GB) and
makes the Phase 5 CI gate affordable. The cost is that CLS pooling and L2
normalisation are implemented here rather than inherited from the reference
implementation, so ``tests/test_embed_parity.py`` checks both backends agree to
1e-4 before any of these vectors are trusted. See DECISIONS.md entry 0014.

Two properties of this model drive the code below:

* Pooling is the **CLS token**, not the mean. ``1_Pooling/config.json`` says
  ``pooling_mode_cls_token: true``. Mean pooling would produce plausible-looking
  vectors that quietly retrieve worse.
* The sequence limit is **512 tokens**, and nothing in the tokenizer enforces
  it (measured: 2,000 words encode to 2,000 tokens). Longer input is truncated
  here, and ``max_content_tokens`` exists so callers can detect that in advance
  rather than discovering it as unexplained recall loss.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

#: [CLS] and [SEP] occupy two of the model's sequence positions.
_SPECIAL_TOKEN_BUDGET = 2


class Embedder(Protocol):
    """Turns text into unit-length vectors."""

    @property
    def dimension(self) -> int: ...

    @property
    def max_content_tokens(self) -> int:
        """Tokens of real text that fit, after reserving space for [CLS]/[SEP]."""
        ...

    def embed_passages(self, texts: Sequence[str]) -> FloatArray: ...

    def embed_query(self, text: str) -> FloatArray: ...


def l2_normalise(vectors: FloatArray) -> FloatArray:
    """Scale each row to unit length.

    bge vectors are compared with cosine similarity, which is only equivalent
    to the inner product once the vectors are normalised.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector cannot be normalised; leave it alone rather than dividing
    # by zero and producing NaNs that would poison every later comparison.
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class OnnxEmbedder:
    """bge-small-en-v1.5 via onnxruntime."""

    def __init__(
        self,
        model_name: str,
        onnx_file: str = "onnx/model.onnx",
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size

        self._max_sequence_length = _read_max_sequence_length(model_name)
        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self._tokenizer.enable_truncation(max_length=self._max_sequence_length)
        self._tokenizer.enable_padding()

        self._session = InferenceSession(
            hf_hub_download(model_name, onnx_file),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        output_shape = self._session.get_outputs()[0].shape
        self._dimension = int(output_shape[-1])

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def max_content_tokens(self) -> int:
        return self._max_sequence_length - _SPECIAL_TOKEN_BUDGET

    def embed_passages(self, texts: Sequence[str]) -> FloatArray:
        """Embed documents. Passages never take the query instruction."""
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        batches = [
            self._embed_batch(list(texts[start : start + self.batch_size]))
            for start in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches).astype(np.float32)

    def embed_query(self, text: str) -> FloatArray:
        """Embed one query, optionally prefixed with the bge instruction."""
        prepared = f"{self.query_instruction}{text}" if self.query_instruction else text
        return self._embed_batch([prepared])

    def _embed_batch(self, texts: list[str]) -> FloatArray:
        encodings = self._tokenizer.encode_batch(texts)
        feeds: dict[str, npt.NDArray[np.int64]] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self._session.run(None, feeds)
        last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
        # CLS pooling: position 0 of every sequence.
        pooled = last_hidden_state[:, 0, :]
        return l2_normalise(pooled)


class SentenceTransformerEmbedder:
    """The reference implementation, used only to validate the ONNX backend.

    Requires the ``ml`` extra (and therefore torch), so nothing on the normal
    path may depend on it.
    """

    def __init__(self, model_name: str, query_instruction: str = "") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 5; support both so the parity test
        # keeps working across versions.
        getter = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_content_tokens(self) -> int:
        return int(self._model.max_seq_length) - _SPECIAL_TOKEN_BUDGET

    def embed_passages(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        encoded = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(encoded, dtype=np.float32)

    def embed_query(self, text: str) -> FloatArray:
        prepared = f"{self.query_instruction}{text}" if self.query_instruction else text
        encoded = self._model.encode([prepared], normalize_embeddings=True)
        return np.asarray(encoded, dtype=np.float32)


def _read_max_sequence_length(model_name: str) -> int:
    """Read the model's real sequence limit instead of assuming 512."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "sentence_bert_config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    limit = config.get("max_seq_length")
    if not isinstance(limit, int) or limit < 8:
        raise ValueError(f"{model_name} reports an implausible max_seq_length: {limit!r}")
    return limit
