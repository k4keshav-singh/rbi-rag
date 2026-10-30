"""Construct the embedder and reranker named by Settings.

Separated from the implementations so the CLI and the API share one place that
decides which backend is in use, and so the import of the heavy modules stays
lazy.
"""

from __future__ import annotations

from rbi_rag.config import Settings
from rbi_rag.index.embed import Embedder, OnnxEmbedder, SentenceTransformerEmbedder
from rbi_rag.retrieve.rerank import OnnxCrossEncoder, Reranker


def build_embedder(settings: Settings, *, for_queries: bool = False) -> Embedder:
    """Build the configured embedder.

    ``for_queries`` decides whether the bge query instruction is applied.
    Passages never take it; the model card is explicit about that.
    """
    instruction = (
        settings.query_instruction if for_queries and settings.use_query_instruction else ""
    )

    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model, query_instruction=instruction)

    return OnnxEmbedder(
        settings.embedding_model,
        onnx_file=settings.onnx_file,
        query_instruction=instruction,
        batch_size=settings.embedding_batch_size,
    )


def build_reranker(settings: Settings) -> Reranker:
    return OnnxCrossEncoder(settings.reranker_model, onnx_file=settings.onnx_file)
