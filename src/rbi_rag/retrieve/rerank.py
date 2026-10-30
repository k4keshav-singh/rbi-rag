"""Cross-encoder reranking with BAAI/bge-reranker-base.

A cross-encoder scores a (query, passage) pair jointly, so it is far more
accurate than comparing two independent embeddings and far too slow to run over
the whole corpus. It reranks the candidates the first-stage retrievers return.

Like the embedder, this runs on onnxruntime so that torch stays out of the
dependency tree. The reranker shares the embedder's 512-token limit, and a
query plus a 512-token chunk will exceed it, so ``truncated_pairs`` records how
often that happened rather than letting it pass unnoticed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rbi_rag.index.embed import FloatArray


class Reranker(Protocol):
    """Scores query/passage pairs. Higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class OnnxCrossEncoder:
    """bge-reranker-base via onnxruntime."""

    def __init__(
        self,
        model_name: str,
        onnx_file: str = "onnx/model.onnx",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncated_pairs = 0

        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()

        self._session = InferenceSession(
            hf_hub_download(model_name, onnx_file),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = list(passages[start : start + self.batch_size])
            scores.extend(self._score_batch(query, batch))
        return scores

    def _score_batch(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        encodings = self._tokenizer.encode_batch(pairs)
        self.truncated_pairs += sum(
            1 for encoding in encodings if len(encoding.ids) >= self.max_length
        )

        feeds: dict[str, FloatArray | np.ndarray[tuple[int, ...], np.dtype[np.int64]]] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        logits = np.asarray(self._session.run(None, feeds)[0], dtype=np.float32)
        # The model emits a single relevance logit per pair.
        return [float(value) for value in logits.reshape(-1)]
