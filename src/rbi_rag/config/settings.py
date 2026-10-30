"""Application configuration.

Every tunable in this project lives here. No other module may read ``os.environ``
directly or hard-code a path, credential, or model name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rbi_rag.schemas import ChunkStrategy, RetrieverName

#: Repository root, derived from this file's location (src/rbi_rag/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EmbeddingBackend = Literal["onnx", "sentence_transformers"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
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
    # Thinking tokens count against max_tokens, so this is not just answer length.
    anthropic_max_tokens: int = Field(default=8192, ge=256)
    # Untuned. Answering from supplied context is not agentic work, so the
    # model default of "high" is probably more than this task needs; Phase 4
    # should sweep it rather than this being left to taste.
    anthropic_effort: Effort = "medium"
    anthropic_timeout_seconds: float = Field(default=120.0, gt=0)
    anthropic_max_retries: int = Field(default=2, ge=0)

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

    # --- Abstention ---
    # Two independent gates; see DECISIONS.md entry 0020.
    #
    # 1. A retrieval gate, compared against the reranker score when reranking
    #    is on and the top dense cosine otherwise. NEVER against an RRF score:
    #    those are ~0.03 for a good match and not comparable to either.
    #    None disables it. It stays None until Phase 4 supplies labelled
    #    unanswerable questions to tune it on -- a threshold invented now would
    #    be a number nobody measured.
    # 2. The model's own `sufficient_evidence` judgement, which always applies.
    abstain_min_score: float | None = None
    abstention_message: str = "Insufficient evidence in the provided circulars."
    # Retries given to the model when it cites something it was not shown.
    citation_retry_attempts: int = Field(default=1, ge=0, le=3)

    # HNSW is approximate. Evaluation uses exact search so that index recall
    # loss is never mistaken for a difference between retrievers.
    exact_search: bool = False

    # --- Logging ---
    log_level: LogLevel = "INFO"

    @field_validator("anthropic_api_key", mode="after")
    @classmethod
    def _blank_key_is_no_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat ANTHROPIC_API_KEY= as absent, not as an empty key.

        .env.example ships the name with no value, so without this the key
        reads as configured and /health reports it as present. The failure then
        surfaces as an authentication error on a user's first real query.
        """
        if value is None or not value.get_secret_value().strip():
            return None
        return value

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
