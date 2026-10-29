# Decisions

One entry per significant tradeoff: context, options, choice, why. Newest last.

---

## 0001 — Postgres with pgvector as the only datastore

**Date:** 2026-10-29 · **Phase:** 0

**Context.** The system needs dense vector search and a keyword/BM25-style
retriever so that hybrid retrieval can be ablated against each one alone.

**Options.**
- A dedicated vector database (Qdrant, Weaviate) plus Elasticsearch or OpenSearch.
- Postgres 16 with `pgvector` for dense search and built-in full-text search
  (`tsvector` + GIN) for keyword search.

**Choice.** Postgres with pgvector, both retrievers in one database.

**Why.** Two engines mean two deployments, two backup stories and two sets of
consistency problems, for a corpus measured in thousands of chunks rather than
hundreds of millions. One database also makes hybrid fusion a single SQL round
trip and keeps the AWS footprint in Phase 6 to one RDS instance. Postgres FTS is
not literally BM25 — `ts_rank_cd` is a different ranking function — and the
evaluation write-up must say so rather than calling it BM25.

---

## 0002 — A shared `schemas/` package

**Date:** 2026-10-29 · **Phase:** 0

**Context.** `Chunk`, `RetrievedChunk` and `Citation` are needed by `ingest`,
`index`, `retrieve`, `generate` and `eval` alike.

**Options.**
- Define each model in the package that first needs it and import across packages.
- One `schemas/` package that every other package imports from.

**Choice.** A shared `schemas/` package.

**Why.** The first option produces import cycles as soon as `eval` needs both a
`Chunk` and a `RetrievedChunk` — by Phase 4 the dependency graph is a knot. A
leaf package that imports nothing from its siblings keeps the graph acyclic by
construction. The cost is one extra package in the tree.

---

## 0003 — Plain SQL migrations instead of Alembic

**Date:** 2026-10-29 · **Phase:** 0

**Context.** The schema needs versioned, repeatable migrations, including
pgvector-specific DDL (HNSW index options) that ORMs model poorly.

**Options.**
- Alembic, which means adding SQLAlchemy to a project with no ORM.
- Numbered `.sql` files applied in order by a small runner that records applied
  versions in a `schema_migrations` table.

**Choice.** Numbered SQL files plus a runner.

**Why.** Alembic's value is autogenerating diffs from ORM models, and there are
no ORM models here — queries are hand-written SQL so that hybrid retrieval and
RRF stay explicit and tunable. Adding SQLAlchemy purely to drive migrations is a
large dependency for little gain, and `CREATE INDEX ... USING hnsw (...) WITH
(m = ..., ef_construction = ...)` would be raw SQL inside Alembic anyway. The
runner is roughly thirty lines. Revisit if the schema outgrows forward-only
migrations.

---

## 0004 — Pin Python to 3.12

**Date:** 2026-10-29 · **Phase:** 0

**Context.** The brief allows 3.11 or newer; the development machine has 3.12.10.

**Options.** Support 3.11+ broadly, or pin a single version matched by CI.

**Choice.** `requires-python = ">=3.12,<3.13"`, and CI resolves the same version.

**Why.** A reproducibility project should not have local and CI resolving
different dependency trees — a lockfile only pins packages, not the interpreter
that determines which wheels those packages resolve to. 3.12 is what is already
installed, and nothing in the stack needs 3.11 compatibility. Widening the range
later is cheap; debugging a 3.11-only CI failure is not.

---

## 0005 — Heavy ML dependencies behind an optional extra

**Date:** 2026-10-29 · **Phase:** 0

**Context.** `sentence-transformers` pulls in `torch`, roughly 2.5 GB on
Windows. It is needed for embedding and reranking, and for nothing else.

**Options.**
- One dependency list, so every `uv sync` and every CI run installs torch.
- A `[project.optional-dependencies] ml` extra installed with `uv sync --extra ml`.

**Choice.** The optional extra.

**Why.** Lint, type-check and unit tests never touch a model. Installing torch
on every CI run would add minutes to every push for no coverage. Everything is
still declared in `pyproject.toml`, so nothing is hidden and the full
environment is one flag away. The Phase 5 CI eval gate runs retrieval-only
against a committed index snapshot and so will need this extra — the workflow
will opt in explicitly at that point.

---

## 0006 — No CLI framework

**Date:** 2026-10-29 · **Phase:** 0

**Context.** Phases 1–4 need command-line entry points for ingest, query, serve
and eval.

**Options.** Add Typer or Click, or use `python -m rbi_rag.<package>` with
`argparse` behind `poe` tasks.

**Choice.** `python -m` plus `argparse`.

**Why.** `poe` already provides task discovery and help text, which is most of
what a CLI framework is bought for. The commands take a handful of flags each.
Rule 8 says not to add dependencies outside the agreed stack without asking, and
this one would not earn its place.

---

## 0007 — Unimplemented tasks fail loudly

**Date:** 2026-10-29 · **Phase:** 0

**Context.** Phase 0 defines `poe` tasks for `ingest`, `query`, `serve` and
`eval`, none of which exist yet.

**Options.** Omit the tasks until their phase lands; have them print a notice
and exit 0; have them exit non-zero.

**Choice.** Exit with status 2 and a message naming the phase.

**Why.** The brief asks for the tasks in Phase 0, so omitting them is out. A
zero exit is worse than useless: it would let a broken pipeline pass a CI step
or a scripted run while reporting success, which is the exact failure mode the
"never invent a result" rule exists to prevent. Exit 2 is unambiguous to both a
human and a shell.
