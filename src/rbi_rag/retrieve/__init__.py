"""Dense, full-text and hybrid retrieval with optional reranking."""

from rbi_rag.retrieve.engine import retrieve
from rbi_rag.retrieve.fuse import fuse_and_order, rank_positions, reciprocal_rank_fusion

__all__ = ["fuse_and_order", "rank_positions", "reciprocal_rank_fusion", "retrieve"]
