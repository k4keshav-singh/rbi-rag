"""Settings must be hermetic: these tests never read the developer's .env."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rbi_rag.config import PROJECT_ROOT, Settings, get_settings


def make_settings(**overrides: object) -> Settings:
    """Build Settings ignoring any .env on disk, so tests are reproducible."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_stable_without_a_dotenv() -> None:
    settings = make_settings()
    assert settings.postgres_port == 5434
    assert settings.postgres_db == "rbi_rag"
    assert settings.embedding_dim == 384
    assert settings.retriever == "hybrid"
    assert settings.rerank_enabled is True


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("RETRIEVER", "dense")
    settings = make_settings()
    assert settings.postgres_port == 6543
    assert settings.retriever == "dense"


def test_dsn_is_assembled_from_parts() -> None:
    settings = make_settings()
    assert settings.postgres_dsn == "postgresql://rbi:rbi@localhost:5434/rbi_rag"


def test_password_is_not_leaked_by_repr() -> None:
    settings = make_settings()
    assert "rbi" not in repr(settings.postgres_password)
    assert settings.postgres_password.get_secret_value() == "rbi"


def test_relative_paths_resolve_against_the_repo_root() -> None:
    settings = make_settings()
    assert settings.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.raw_dir.is_absolute()


def test_absolute_paths_are_left_alone() -> None:
    absolute = Path(PROJECT_ROOT).resolve() / "elsewhere"
    settings = make_settings(data_dir=absolute)
    assert settings.raw_dir == absolute / "raw"


def test_crawl_delay_floor_is_enforced() -> None:
    """Politeness toward rbi.org.in is a validation rule, not a convention."""
    with pytest.raises(ValidationError):
        make_settings(download_delay_seconds=0.1)


def test_unknown_retriever_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(retriever="magic")


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_a_blank_api_key_counts_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """ANTHROPIC_API_KEY= would otherwise read as a configured key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert make_settings().anthropic_api_key is None


def test_a_whitespace_api_key_counts_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert make_settings().anthropic_api_key is None


def test_a_real_api_key_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-example")
    key = make_settings().anthropic_api_key
    assert key is not None
    assert key.get_secret_value() == "sk-ant-example"
