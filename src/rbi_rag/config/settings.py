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

#: Repository root, derived from this file's location (src/rbi_rag/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RetrieverName = Literal["dense", "fts", "hybrid"]
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
    postgres_port: int = 5432
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

    # --- Ingestion ---
    download_delay_seconds: float = Field(
        default=2.0,
        ge=2.0,
        description="Minimum seconds between requests to rbi.org.in. Floor is deliberate.",
    )
    download_max_retries: int = Field(default=5, ge=0)
    user_agent: str = "rbi-rag/0.1"

    # --- Paths ---
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    # --- Retrieval defaults (overridable per request and per ablation run) ---
    retriever: RetrieverName = "hybrid"
    top_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_enabled: bool = True
    abstain_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

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
