"""Forward-only SQL migrations.

Numbered ``.sql`` files in ``migrations/`` are applied in filename order and
recorded in ``schema_migrations``, so re-running is a no-op. Deliberately not
Alembic: there are no ORM models to diff against, and pgvector's index DDL
would be raw SQL inside Alembic anyway. See DECISIONS.md entry 0003.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: The vector column width baked into 001_initial.sql.
SCHEMA_EMBEDDING_DIM = 384

_CREATE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """A migration could not be applied."""


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Return every migration, ordered by filename."""
    return [
        Migration(version=path.stem, path=path) for path in sorted(directory.glob("[0-9]*.sql"))
    ]


def applied_versions(connection: psycopg.Connection[tuple[str, ...]]) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(_CREATE_TRACKING_TABLE)
        cursor.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cursor.fetchall()}


def migrate(
    connection: psycopg.Connection[tuple[str, ...]],
    directory: Path = MIGRATIONS_DIR,
    embedding_dim: int = SCHEMA_EMBEDDING_DIM,
) -> list[str]:
    """Apply outstanding migrations. Returns the versions applied, if any."""
    if embedding_dim != SCHEMA_EMBEDDING_DIM:
        raise MigrationError(
            f"EMBEDDING_DIM is {embedding_dim} but the schema declares "
            f"vector({SCHEMA_EMBEDDING_DIM}). Changing the embedding model needs a new "
            f"migration that alters the column, not a settings change alone."
        )

    already = applied_versions(connection)
    newly_applied: list[str] = []

    for migration in discover(directory):
        if migration.version in already:
            continue
        with connection.cursor() as cursor:
            cursor.execute(migration.sql)
            cursor.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (migration.version,)
            )
        connection.commit()
        newly_applied.append(migration.version)

    return newly_applied
