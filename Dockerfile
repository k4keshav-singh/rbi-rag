# syntax=docker/dockerfile:1
#
# The embedding model is baked in at build time rather than downloaded on
# first request. A cold start that reaches out to HuggingFace is slow, can be
# rate-limited, and fails outright in a subnet with no egress -- all of which
# would surface as an opaque timeout on a user's first question.
#
# The reranker is NOT baked in: its ONNX export is 1061 MB on disk and 1.3 GB
# resident (measured), against 127 MB and 187 MB for the embedder. Including
# it would rule out every free-tier instance size. Deployments that want
# reranking set RERANK_ENABLED=true and need ~2 GB of RAM; the model then
# downloads on first use. See DECISIONS.md entry 0034.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so a source change does not re-resolve the whole tree.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Fetch exactly the embedding-model files the ONNX embedder opens.
ENV HF_HOME=/opt/hf
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run python -c "\
from huggingface_hub import hf_hub_download; \
m='BAAI/bge-small-en-v1.5'; \
[hf_hub_download(m, f) for f in ('onnx/model.onnx','tokenizer.json','sentence_bert_config.json','config.json')]"


FROM python:3.12-slim AS runtime

# curl is for the container healthcheck and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /opt/hf /opt/hf

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    RERANK_ENABLED=false \
    LOG_LEVEL=INFO

USER app
EXPOSE 8000

# Reports degraded rather than failing when a dependency is missing, so the
# container stays up and /health can say what is wrong.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "rbi_rag.api", "--host", "0.0.0.0", "--port", "8000"]
