# Migrations

Numbered, forward-only SQL files applied in filename order by the runner in
`rbi_rag.index` (Phase 2). Applied versions are recorded in a `schema_migrations`
table so that re-running is a no-op.

Naming: `NNN_short_description.sql`, starting at `001`.

Deliberately plain SQL rather than Alembic — see DECISIONS.md entry 0003.
