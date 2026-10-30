# CLAUDE.md — conventions for this repository

Read this before changing anything.

## What this project is

A question-answering system over public RBI (Reserve Bank of India) Master
Directions that returns answers with verifiable citations — circular number,
page, source URL — backed by an evaluation suite and an ablation study.

This is a portfolio project. **Code quality, honest measurement and clear
documentation matter more than feature count.**

## Phase gating

Work happens one phase at a time. For each phase:

1. Present a short plan and wait for approval.
2. Implement.
3. Run tests and lint, and show the real output.
4. Summarise what changed, then **stop**.

Do not begin the next phase until asked.

| Phase | Scope | Status |
| ----- | ----------------------------- | ----------- |
| 0 | Scaffold | done |
| 1 | Ingestion | done |
| 2 | Indexing and retrieval | done |
| 3 | Generation API | done |
| 4 | Evaluation | done |
| 5 | CI eval gate | done |
| 6 | Frontend and AWS | done |
| 7 | Documentation | not started |

## Hard rules

1. **Never invent a number.** Every metric, benchmark or result in `README.md`
   or `results/` must come from a script in this repository that was actually
   run. If it has not been measured, write `TODO`. This rule has no exceptions.
2. **Never mark an eval item `verified_by_human`.** A drafting script may
   propose candidate questions, always with `verified_by_human=false`. Only the
   repository owner flips that flag, and all reported metrics use verified
   items only.
3. **Type hints everywhere.** `mypy --strict` must pass on `src/`.
4. **Tests for every module.** Network calls and LLM calls are mocked in unit
   tests. No test may touch the network.
5. **Config through `pydantic-settings` only.** No hard-coded paths, keys,
   model names or thresholds outside `src/rbi_rag/config/settings.py`.
6. **Architecture decisions get a `DECISIONS.md` entry** — context, options,
   choice, rationale.
7. **Ask instead of guessing** when something is ambiguous.
8. **Do not add dependencies** outside the agreed stack without asking first.
9. **Never run `terraform apply`/`destroy`** or any AWS CLI command that
   creates, changes or deletes resources. Write the command; the owner runs it.
10. **Commits are small and conventional** (`feat:`, `fix:`, `docs:`,
    `chore:`, `test:`, `refactor:`). The owner reviews and pushes.

## Environment

- Windows, **PowerShell**. Every command and script must work there. No
  Makefiles, no bash-only scripts. Task running is `poethepoet` via
  `uv run poe <task>`.
- Python 3.12, managed by `uv`.
- Docker Desktop (WSL2) for Postgres.
- Secrets live only in `.env` (git-ignored). `.env.example` is committed and
  lists every key with no real values.

## Stack

Chosen deliberately; do not swap a component without a `DECISIONS.md` entry.

- **FastAPI + Pydantic v2** — typed API and schemas.
- **PostgreSQL 16 + pgvector** — one database for dense vectors *and* keyword
  search (Postgres full-text search with `tsvector` as the BM25-style retriever).
- **Embeddings** — `BAAI/bge-small-en-v1.5`, local, run on **onnxruntime**
  rather than sentence-transformers (DECISIONS.md 0014). torch is never
  installed outside the `ml` extra, which exists only for the parity test.
- **Reranker** — `BAAI/bge-reranker-base` cross-encoder, local, also ONNX.
- **Fusion** — Reciprocal Rank Fusion.
- **LLM** — Anthropic API, model from `ANTHROPIC_MODEL`.
- **PDF parsing** — PyMuPDF.
- **Tooling** — ruff, mypy, pytest, pre-commit, GitHub Actions.
- **IaC** — Terraform, Phase 6 only.

## Layout

```
src/rbi_rag/
  config/      settings, the only place that reads the environment
  schemas/     Pydantic models shared by every other package
  ingest/      downloader, parser, cleaner, chunker
  index/       schema, migrations, embedding storage
  retrieve/    dense, FTS, hybrid, rerank
  generate/    prompting, structured output, citation validation
  api/         FastAPI app
  eval/        metrics, judge, ablation runner
tests/         mirrors src/; fixtures in tests/fixtures/
eval/          evaluation sets (JSONL)
results/       generated tables — never hand-edited
infra/         Terraform (Phase 6)
docs/          design notes, diagrams
data/          git-ignored; re-creatable with `uv run poe ingest`
```

`schemas/` exists so no two packages import each other just to agree on a type.
New shared models go there, not into whichever package happened to need one first.

## Common commands

```powershell
uv sync                  # install dependencies
uv run poe lint          # ruff check + ruff format --check + mypy
uv run poe fmt           # auto-fix and format
uv run poe test          # unit tests only (no db, no network, no models)
uv run poe test-integration  # needs db-up
uv run poe test-model    # downloads models
uv run poe test-parity   # needs: uv sync --extra ml
uv run poe db-up         # start Postgres, wait for healthy
uv run poe db-down       # stop Postgres, keep data
uv run poe db-nuke       # stop Postgres and delete the volume
```

```powershell
uv run poe ingest harvest   # listing -> data/manifest.jsonl
uv run poe ingest urls      # PDFs still to download
uv run poe ingest status    # hash data/raw, report gaps
uv run poe ingest process   # parse, clean, chunk
```

```powershell
uv run poe index migrate    # apply SQL migrations
uv run poe index build      # embed and index a chunk set
uv run poe index sets       # list chunk sets
uv run poe query "..."      # ranked chunks, with per-retriever provenance
```

```powershell
uv run poe eval status      # eval set summary
uv run poe eval review      # verify candidates (only the owner may)
uv run poe eval retrieval   # full 36-arm sweep, no LLM
uv run poe eval             # retrieval, then generation on a subset
uv run poe eval gate        # CI regression gate on the committed snapshot
uv run poe eval snapshot    # rebuild that snapshot / re-record its baseline
```

## Measurement rules that bind later phases

- Evaluation retrieves with `RetrievalConfig.exact = True`. HNSW is
  approximate, and its recall loss must never be folded into a comparison
  between retrievers.
- The keyword arm is called **FTS**, never BM25. `ts_rank_cd` is a different
  ranking function.
- Chunks longer than the embedding window (510 content tokens) are truncated
  when embedded. `poe index build` counts them; any result involving
  1024-token chunks must report that count beside it.
- Abstention thresholds and `ANTHROPIC_EFFORT` are untuned and must be swept
  in Phase 4, not guessed. `abstain_min_score` is deliberately `None`.
- `cost_usd` is arithmetic over a dated price table, never an estimate. An
  unpriced model reports `None`.
- A citation whose quote is not verbatim in its chunk is a failure, not a
  near miss. The pipeline abstains rather than stripping bad citations.
- Gold evidence is labelled as spans (doc, pages, quote), never as chunk
  ids: ids do not survive a re-chunk, so a chunk-id gold set is valid for
  one ablation arm and empty for the rest.
- Recall counts gold spans covered, not chunks retrieved. Chunk counting
  favours large chunks for reasons unrelated to retrieval quality.
- Faithfulness is never reported without its judge calibration beside it.
- The in-memory store and Postgres must agree on matching semantics. They
  once did not (OR vs AND) and unit tests passed on a broken retriever.
- The CI gate measures change on a fixture corpus, never quality. Its
  numbers do not belong in README or results/.
- Reranking is off in deployment until the ablation shows it helps. It
  costs 1.32 GB of RAM (measured) against the API's 242 MB without it.
- Never gate on recall@k alone. With a snapshot smaller than k it cannot
  move: inverting the dense retriever left recall@5 at 1.0000 while MRR
  fell to 0.25. Gate order-sensitive metrics too.

## Working with RBI's website

The downloader must respect `robots.txt`, wait at least two seconds between
requests (enforced in `Settings`, not by convention), retry with backoff, and be
idempotent via content hashing. Re-running a full ingest must be a no-op.
