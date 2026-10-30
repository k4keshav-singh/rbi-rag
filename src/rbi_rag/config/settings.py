"""Application configuration.

Every tunable in this project lives here. No other module may read ``os.environ``
directly or hard-code a path, credential, or model name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rbi_rag.schemas import ChunkStrategy, RetrieverName

#: Repository root, derived from this file's location (src/rbi_rag/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EmbeddingBackend = Literal["onnx", "sentence_transformers"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Settings resolved from the environment, then ``.env``, then these defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Postgres ---
    postgres_host: str = "localhost"
    # 5434, not 5432: a natively installed Postgres commonly occupies 5432 and
    # would answer instead of the container, failing auth in a confusing way.
    postgres_port: int = 5434
    postgres_user: str = "rbi"
    postgres_password: SecretStr = SecretStr("rbi")
    postgres_db: str = "rbi_rag"

    # --- Anthropic ---
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"

    # --- Local models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_backend: EmbeddingBackend = "onnx"
    onnx_file: str = Field(
        default="onnx/model.onnx",
        description="Path of the ONNX export inside the model repository.",
    )
    embedding_batch_size: int = Field(default=32, ge=1)

    # bge-*-v1.5 works well without an instruction; the model card says to pick
    # whichever performs better on the task, so this is an ablation dimension.
    use_query_instruction: bool = False
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Ingestion ---
    download_delay_seconds: float = Field(
        default=2.0,
        ge=2.0,
        description="Minimum seconds between requests to rbi.org.in. Floor is deliberate.",
    )
    download_max_retries: int = Field(default=5, ge=0)
    user_agent: str = "rbi-rag/0.1"
    listing_url: str = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"

    # --- Chunking (the ablation sweeps these; these are the defaults) ---
    chunk_strategy: ChunkStrategy = "section_aware"
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)

    # --- Paths ---
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    # --- Retrieval defaults (overridable per request and per ablation run) ---
    retriever: RetrieverName = "hybrid"
    top_k: int = Field(default=10, ge=1)
    candidate_k: int = Field(
        default=50, ge=1, description="Per-retriever depth fetched before fusion."
    )
    rrf_k: int = Field(default=60, ge=1)
    rerank_enabled: bool = True
    rerank_candidates: int = Field(default=50, ge=1)
    abstain_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # HNSW is approximate. Evaluation uses exact search so that index recall
    # loss is never mistaken for a difference between retrievers.
    exact_search: bool = False

    # --- Logging ---
    log_level: LogLevel = "INFO"

    @property
    def postgres_dsn(self) -> str:
        """libpq connection string, including the password."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def raw_dir(self) -> Path:
        """Where downloaded PDFs land, as an absolute path."""
        return self._absolute(self.data_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        """Where parsed pages and chunk files are written."""
        return self._absolute(self.data_dir) / "processed"

    @property
    def manifest_path(self) -> Path:
        """The ingest manifest, one JSONL row per Master Direction."""
        return self._absolute(self.data_dir) / "manifest.jsonl"

    @property
    def results_path(self) -> Path:
        """Where ablation tables are written, as an absolute path."""
        return self._absolute(self.results_dir)

    @staticmethod
    def _absolute(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
