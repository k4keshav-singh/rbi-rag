# Reproducing this project

From a clean clone to a results table. Three steps need a human and cannot be
automated; they are marked **MANUAL** and the reason is given each time.

Everything is PowerShell. There are no Makefiles and no bash-only scripts.

---

## 0. Prerequisites

- Windows with PowerShell, [uv](https://docs.astral.sh/uv/), Docker Desktop
  (WSL2 backend)
- An Anthropic API key, for generation and evaluation only — ingestion,
  indexing and retrieval need none

```powershell
uv sync
Copy-Item .env.example .env
uv run pre-commit install
```

Open `.env` and set `ANTHROPIC_API_KEY`.

> If something owns port 5432 already — a locally installed PostgreSQL is
> common — leave `POSTGRES_PORT=5434` as it ships. The container and the DSN
> read the same variable, so they cannot disagree.

```powershell
uv run poe db-up
uv run poe lint
uv run poe test
```

## 1. Harvest the catalogue

One rate-limited request to rbi.org.in. Re-running is a no-op.

```powershell
uv run poe ingest harvest
```

Expect roughly 380 entries, each with a title, date, category and PDF URL.

## 2. Acquire the PDFs — **MANUAL**

```powershell
uv run poe ingest urls      # writes data/pdf_urls.txt
```

RBI serves its PDFs from a host that answers scripted clients with a
JavaScript bot challenge. This project does not solve, replay or imitate its
way past that. Open the URLs in a browser and save the files into `data/raw/`,
keeping their original names.

Twenty to thirty documents is enough to make the evaluation meaningful.

```powershell
uv run poe ingest status
```

`status` hashes what arrived and flags any file that is not really a PDF — a
browser that hit the challenge saves an HTML page under a `.PDF` name, and
that would otherwise parse to an empty document.

## 3. Parse, clean and chunk

```powershell
uv run poe ingest process
```

For the ablation, build every chunking configuration:

```powershell
foreach ($s in "fixed_tokens","section_aware") { foreach ($n in 256,512,1024) { uv run poe ingest process --strategy $s --size $n } }
```

## 4. Index

```powershell
uv run poe index migrate
```

```powershell
foreach ($s in "fixed_tokens","section_aware") { foreach ($n in 256,512,1024) { uv run poe index build --strategy $s --size $n } }
```

```powershell
uv run poe index sets
```

`index build` reports how many chunks exceed the embedding window and were
truncated. Any result involving 1024-token chunks has to carry that count
beside it.

## 5. Ask something

```powershell
uv run poe query "what is the minimum capital adequacy ratio"
```

The output shows which first-stage retriever found each chunk and at what
rank, so a ranking can be explained rather than only trusted.

```powershell
uv run poe serve
```

Then open <http://127.0.0.1:8000/>.

## 6. Build an evaluation set — **MANUAL**

```powershell
uv run poe eval draft --chunk-set 1 --limit 40
```

Drafted questions are written with `verified_by_human=false`. They are
suggestions, not ground truth: grading the system against questions its own
model wrote would measure nothing.

```powershell
uv run poe eval review
```

Each candidate appears with the source text behind its gold span. Keep, edit,
reject or skip. **This is the only path in the repository that can mark an
item verified**, and every reported metric filters on that flag.

```powershell
uv run poe eval status
```

## 7. Run the evaluation

```powershell
uv run poe eval retrieval      # all 36 arms, no LLM, free
```

```powershell
uv run poe eval                # retrieval, then generation on a subset
```

Writes `results/retrieval.{md,csv}` and `results/generation.{md,csv}` with the
git SHA, the date and the size of the verified set.

Retrieval figures reproduce exactly. Generation figures replay from a
content-hash cache; clearing it re-samples the model, which the results header
says rather than implying determinism it does not have.

## 8. Calibrate the judge — **MANUAL**

Faithfulness is scored by an LLM. Most answers in a working system are
faithful, so a judge that always said "faithful" would score around 90% raw
agreement and be worthless. Label a subset yourself and the report prints
Cohen's kappa beside the faithfulness score; until then it prints `TODO`.

```powershell
uv run poe eval calibrate
```

## 9. Update the documentation

```powershell
uv run poe docs
```

Regenerates the README results table from `results/*.csv` and the DECISIONS
index from its entries. Numbers reach the README by being measured or they do
not reach it at all.

## 10. The CI gate

```powershell
uv run poe eval gate
```

Compares retrieval on the committed fixture snapshot against the committed
baseline. It detects change, not quality. After an intended change:

```powershell
uv run poe eval snapshot --baseline-only
```

and say why in the commit message.

## 11. Deploy

See [infra/README.md](../infra/README.md) for the deployment commands, the
cost table and a teardown guide. The Terraform validates but **has never been
applied**.

---

## What is reproducible, and what is not

| Step | Reproducible |
| --- | --- |
| Harvest | yes — re-running is a no-op |
| Parse, clean, chunk | yes — byte-identical from the same PDFs |
| Index | yes — vectors are deterministic |
| Retrieval metrics | yes — exact kNN, no sampling |
| Generation metrics | replayed from cache; re-sampled if cleared |
| PDF acquisition | no — manual, and RBI may reissue a document |
| Eval set | no — it encodes your judgement, which is the point |
