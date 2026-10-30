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

---

## 0008 — Browser-assisted PDF acquisition

**Date:** 2026-10-29 · **Phase:** 1

**Context.** RBI serves Master Direction PDFs from `rbidocs.rbi.org.in`, which
answers scripted clients with an F5 BIG-IP JavaScript challenge page carrying a
200 status and `Content-Type: text/html`. Measured during Phase 1 planning:
ordinary cookie handling does not satisfy it across repeated requests. The
listing host, `www.rbi.org.in`, has no such protection and serves 788 KB of
real HTML to a plain client.

**Options.**
- Solve the challenge: execute its JavaScript, or drive a headless browser.
- Use a TLS/HTTP client that imitates a browser fingerprint.
- Separate acquisition from ingestion: harvest metadata by script, obtain PDFs
  through a real browser, and process from a local raw store.
- Abandon the RBI corpus.

**Choice.** Separate acquisition from ingestion.

**Why.** The first two options defeat an access control the publisher put there
deliberately, and the fact that the documents are public does not make
circumventing it acceptable. The third gives up nothing that matters: the
manifest, the content hashing, the idempotency guarantee and every downstream
stage are unchanged, because they were always going to read from `data/raw/`.
What is lost is unattended end-to-end automation, which costs one manual step
per corpus refresh.

The seam is the `Fetcher` protocol in `ingest/fetch.py`. `LocalFetcher` reads
the raw store; `BlockedHttpFetcher` exists solely to raise with an explanation,
so the reason is discoverable from the code and not only from this file. If RBI
ever permits scripted access, one implementation replaces the other.

One guard earns its place: a browser that hits the challenge saves an HTML page
under a `.PDF` name. `looks_like_pdf` checks the magic bytes, because such a
file would otherwise parse to an empty document and silently shrink the corpus.

---

## 0009 — Treating an unreachable robots.txt as permissive

**Date:** 2026-10-29 · **Phase:** 1

**Context.** The brief requires respecting `robots.txt`. RBI returns HTTP 418
for `https://www.rbi.org.in/robots.txt` to every scripted client tried, so its
policy cannot be read at all.

**Options.** Treat any failure as a full disallow and stop; ignore robots
entirely; or follow the status-code semantics the standard defines.

**Choice.** Follow RFC 9309 §2.3.1: a 4xx means no policy is published and
access is unrestricted; a 5xx or a network failure means assume a complete
disallow. 418 is a 4xx, so harvesting proceeds.

**Why.** This is the behaviour the standard specifies, not a convenience. The
important part is honesty about it: `PoliteClient.robots_decision` returns the
reason alongside the verdict, and the harvest report prints it, so every run
records that no policy was actually read rather than implying one was honoured.
The two-second floor between requests is enforced in `Settings` regardless.

---

## 0010 — Harvesting metadata from the listing page alone

**Date:** 2026-10-29 · **Phase:** 1

**Context.** The manifest needs url, title, circular number, date and
department. The listing page carries title, date, category and the PDF link.
Each entry also links to a detail page, and RBI prints the circular number and
department on page 1 of every Master Direction.

**Options.** Fetch one detail page per Direction (~380 extra requests), or take
the identifiers off the PDF during parsing.

**Choice.** Read them from the PDF; never fetch detail pages.

**Why.** It turns 380 requests into one, which matters when the standing rate
limit is two seconds. The detail pages are also heavily JavaScript-driven,
making them a worse parsing target than the PDF's own first page. The cost is
that circular number and department are unknown until a PDF is acquired, which
is why `merge_harvested` preserves PDF-derived fields across a re-harvest
rather than letting the listing overwrite them with nulls.

---

## 0011 — Block-based PDF text extraction

**Date:** 2026-10-29 · **Phase:** 1

**Context.** The chunker packs paragraphs up to a token budget, so it needs to
know where paragraphs end.

**Options.** PyMuPDF's flat `get_text("text")`, or `get_text("blocks")` joined
with a blank line.

**Choice.** Blocks.

**Why.** Flat extraction separates every line with a single newline and gives
no paragraph boundary at all, so an entire page collapses into one unit and
"chunking" degenerates into fixed-width slicing of a page. That would have made
the `section_aware` versus `fixed_tokens` comparison in Phase 4 meaningless,
since both arms would have been doing the same thing. Blocks correspond to
paragraphs and are already in reading order once sorted by position.

---

## 0012 — Chunking by accumulating paragraphs, not by slicing tokens

**Date:** 2026-10-29 · **Phase:** 1

**Context.** A chunker can either tokenise the document and slice fixed windows
of token ids, decoding each back to text, or accumulate whole paragraphs until
the budget is reached.

**Options.** As above.

**Choice.** Accumulate paragraphs.

**Why.** Decoding token ids back to text is lossy: it normalises whitespace and
can alter punctuation, so the stored chunk would differ from the source it
points at. This system's entire claim is that a citation can be checked against
the PDF, and a quote that does not appear in the document breaks that claim.
A test asserts every chunk's paragraphs appear verbatim in the source text.

The cost is that chunks land under the budget rather than exactly on it, and
that an oversized paragraph needs splitting on sentences and then on words.

---

## 0013 — Adding `tokenizers` as a direct dependency

**Date:** 2026-10-29 · **Phase:** 1

**Context.** Chunk size is the independent variable of the Phase 4 ablation, so
it must be counted with the tokenizer the embedding model actually uses.
`sentence-transformers` provides one but lives behind the optional `ml` extra
(entry 0005), and pulling torch in just to count tokens would undo that.

**Options.** Approximate with a whitespace or character heuristic; require the
`ml` extra for chunking; or depend on `tokenizers` directly.

**Choice.** `tokenizers` as a base dependency, behind a `TokenCounter` Protocol.

**Why.** An approximation is not defensible when the number being approximated
*is* the variable under study. `tokenizers` is a small Rust binding with no
torch dependency, and it is already a transitive dependency of
`sentence-transformers`, so it adds nothing to the eventual full install. The
Protocol keeps tests offline: they inject `WhitespaceTokenCounter`, which warns
loudly that its counts must never be reported.

**Measured, not assumed.** The tokenizer does not enforce the model's sequence
limit: `BAAI/bge-small-en-v1.5` encoded 2,000 words to 2,000 tokens with no
truncation configured. The 512-token cap belongs to the model, so a 1024-token
chunk will be truncated silently inside `sentence-transformers` at embed time.
Phase 2 must assert the model's real `max_seq_length` and refuse, or loudly
record, any chunk that exceeds it. See the note in README under Limitations.

---

## 0014 — onnxruntime as the embedding backend, not sentence-transformers

**Date:** 2026-10-30 · **Phase:** 2

**Context.** The agreed stack names sentence-transformers, which means torch:
roughly 2.5 GB on Windows. Phase 5 runs a retrieval eval in CI on every push,
so that install cost recurs forever. Both `bge-small-en-v1.5` and
`bge-reranker-base` publish ONNX exports under an MIT licence, and
`onnxruntime` was already present in the dependency tree as a PyMuPDF
transitive dependency.

**Options.**
- sentence-transformers, as specified: the reference implementation, zero
  correctness risk, 2.5 GB.
- onnxruntime: ~130 MB, CPU-fast, but pooling and normalisation have to be
  implemented here rather than inherited.
- Both: ONNX by default, sentence-transformers retained purely to validate it.

**Choice.** The third. ONNX is the default; sentence-transformers stays in the
`ml` extra and is used by nothing except `tests/test_embed_parity.py`.

**Why.** The risk of hand-rolling pooling is real and specific: bge uses
**CLS** pooling, not mean pooling, and getting that wrong produces vectors that
look entirely plausible while retrieving worse. That is not a risk to argue
away, so it is measured instead.

**Measured.** Over the parity sentence set, against sentence-transformers:

| Quantity | Value |
| --- | --- |
| Max elementwise difference | 1.639e-07 |
| Mean elementwise difference | 3.202e-08 |
| Minimum cosine similarity | 1.00000000 |
| Dimension (ONNX / reference) | 384 / 384 |
| Content-token limit (ONNX / reference) | 510 / 510 |

That is float32 rounding, not an implementation difference. The parity suite
also asserts the two backends rank the same sentences in the same order, since
a difference too small to see could still reorder results.

The parity test is excluded from the default suite and from CI, because
installing torch on every push is the cost this decision exists to avoid. Run
it with `uv sync --extra ml` and `uv run poe test-parity` whenever the embedder
changes.

This adds three direct dependencies — `onnxruntime`, `huggingface-hub` and
`numpy` — all of which were already installed transitively, so the resolved
environment is unchanged. They are declared explicitly because depending on
another package's transitive dependencies is fragile.

---

## 0015 — Postgres full-text search as the keyword retriever

**Date:** 2026-10-30 · **Phase:** 2

**Context.** The hybrid retriever needs a lexical counterpart to dense search.
The stack puts both in one database.

**Options.** A separate search engine with real BM25, or Postgres full-text
search with a GIN-indexed `tsvector`.

**Choice.** Postgres FTS: a generated `tsvector` column using the `english`
configuration, queried with `websearch_to_tsquery` and ranked by `ts_rank_cd`.

**Why.** Entry 0001 already committed to one datastore. `websearch_to_tsquery`
accepts quoted phrases and negation without throwing on input a user might
reasonably type, which `to_tsquery` does not. The column is `GENERATED ALWAYS`
so the index can never drift out of step with the text it indexes.

**The caveat that must be reported.** `ts_rank_cd` is **not BM25**. It has no
document-length normalisation of the kind BM25 applies, and no tunable k1/b.
Calling this arm "BM25" in the results would be wrong, so it is called "FTS"
everywhere. The `english` configuration also applies Snowball stemming, which
is right for ordinary prose but conflates regulatory terms a lawyer would
distinguish. If this arm underperforms, that is a candidate explanation and
should be investigated rather than reported as a dense-versus-sparse finding.

---

## 0016 — Evaluation searches exactly; HNSW serves queries

**Date:** 2026-10-30 · **Phase:** 2

**Context.** HNSW is an approximate index: it trades recall for speed. The
Phase 4 ablation compares retrievers on recall@k.

**Options.** Measure through HNSW like production does, or measure with exact
kNN and treat the index as a serving concern.

**Choice.** `RetrievalConfig.exact` forces a sequential scan, evaluation sets
it, and HNSW-versus-exact agreement is reported as its own number.

**Why.** Running eval queries through an approximate index mixes the index's
own recall loss into the retriever comparison, where it is indistinguishable
from a real effect. With a corpus of roughly 19,000 chunks an exact scan is
cheap, so there is no reason to accept that contamination. The index still
exists and is still measured — just not as a hidden term in someone else's
result. An integration test asserts the two orderings agree on the fixture
corpus; if they ever diverge, every number measured through HNSW is suspect.

---

## 0017 — One partial HNSW index per chunk set

**Date:** 2026-10-30 · **Phase:** 2

**Context.** The ablation sweeps chunking strategy against chunk size, so
several chunk sets live in the `chunks` table simultaneously and each query
must see exactly one of them.

**Options.** A single HNSW index over every row with a `chunk_set_id` filter,
pgvector 0.8's `hnsw.iterative_scan`, or a partial index per set.

**Choice.** A partial index per set, created by the index builder.

**Why.** A shared index forces post-filtering: HNSW returns its nearest
neighbours across all sets and the filter then discards most of them, so the
effective recall for any one set is far below what the index would give on its
own. That is the problem in entry 0016 wearing a different hat. A partial index
contains only its own set, so no filtering is needed. Six configurations means
six small indexes, which costs build time and nothing else. An integration test
asserts the index definition really is restricted to its chunk set.

---

## 0018 — Reciprocal Rank Fusion in Python, not SQL

**Date:** 2026-10-30 · **Phase:** 2

**Context.** Hybrid retrieval fuses a dense ranking with a keyword ranking.
Both come from the same database, so the fusion could be a single SQL query.

**Options.** One SQL statement with CTEs, or two queries fused in Python.

**Choice.** Two queries, fused in `retrieve/fuse.py`.

**Why.** The Phase 2 done-criterion is that every retriever is unit-tested and
switchable purely by config. Fusion in Python is a pure function over lists of
ids: its tests need no database, no fixtures and no Postgres running, and they
can assert the property that justifies RRF at all — that a chunk both
retrievers rank beats a chunk only one of them ranks. The same logic in SQL
would only be testable through an integration test.

It also avoids inventing a normalisation. Dense scores are cosine similarities
in [-1, 1]; `ts_rank_cd` is unbounded and scaled differently. RRF uses only
rank positions, so the two never need to be made commensurable. The cost is one
extra round trip per hybrid query.

---

## 0019 — The container's Postgres listens on 5434

**Date:** 2026-10-30 · **Phase:** 2

**Context.** Integration tests failed with `password authentication failed for
user "rbi"` while the container reported healthy and `docker compose exec psql`
worked. A locally installed PostgreSQL 18 service was listening on 5432 and
answering instead of the container. 5433 was then held by a stale Docker
binding.

**Options.** Ask the developer to stop their local Postgres, or move the
container off the contested port.

**Choice.** Default `POSTGRES_PORT` to 5434, used by both `docker-compose.yml`
and the DSN in `Settings`, so the two cannot disagree.

**Why.** A natively installed Postgres on 5432 is common on developer machines,
and this failure mode is unusually confusing: the container is healthy, the
credentials are right, and the error says the password is wrong. Stopping
someone's local database to run a side project is the wrong trade. Changing one
setting is not.
