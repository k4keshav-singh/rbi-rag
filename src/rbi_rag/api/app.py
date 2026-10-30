"""The FastAPI application."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from rbi_rag.api.deps import (
    AppState,
    StoreProvider,
    default_chunk_set_resolver,
    postgres_store_provider,
)
from rbi_rag.config import Settings, get_settings
from rbi_rag.generate.client import GenerationError, LLMClient
from rbi_rag.generate.cost import PRICE_TABLE_DATE, price_for
from rbi_rag.generate.pipeline import answer_question
from rbi_rag.index.embed import Embedder
from rbi_rag.retrieve.rerank import Reranker
from rbi_rag.schemas import QueryRequest, QueryResponse, RetrievalConfig

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    store_provider: StoreProvider | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chunk_set_resolver: Callable[[], int] | None = None,
) -> FastAPI:
    """Build the app. Every dependency can be supplied, which is how it is tested."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(
            settings=resolved,
            store_provider=store_provider or postgres_store_provider(resolved),
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            chunk_set_resolver=chunk_set_resolver or default_chunk_set_resolver(resolved),
        )
        # Built once: an ONNX session per request would dominate latency.
        if state.embedder is None and embedder is None:
            from rbi_rag.index.factory import build_embedder

            state.embedder = build_embedder(resolved, for_queries=True)
        if state.reranker is None and reranker is None and resolved.rerank_enabled:
            from rbi_rag.index.factory import build_reranker

            state.reranker = build_reranker(resolved)
        if state.llm is None and llm is None:
            from rbi_rag.generate.client import AnthropicClient, GenerationError

            try:
                state.llm = AnthropicClient(resolved)
            except GenerationError as exc:
                # Start anyway, so /health can say what is missing. /query
                # returns 503 until it is fixed. Refusing to boot would hide
                # the reason behind a crashed process.
                logger.warning("LLM client unavailable", extra={"reason": str(exc)})
        app.state.deps = state
        yield

    app = FastAPI(
        title="rbi-rag",
        version="0.1.0",
        summary="Question answering over RBI Master Directions, with verifiable citations.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    @app.get("/health")
    def health() -> JSONResponse:
        """Readiness, with each dependency reported separately.

        A degraded response still returns 200 with the detail: a monitor should
        be able to see *which* dependency is down, not just that something is.
        """
        state: AppState = app.state.deps
        checks: dict[str, Any] = {
            "api_key_configured": state.settings.anthropic_api_key is not None,
            "model": state.settings.anthropic_model,
            "price_table_date": PRICE_TABLE_DATE,
            "model_priced": price_for(state.settings.anthropic_model) is not None,
            "embedder_loaded": state.embedder is not None,
            "llm_ready": state.llm is not None,
        }
        try:
            with state.store_provider():
                checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"unavailable: {exc}"

        healthy = checks["database"] == "ok" and checks["llm_ready"]
        return JSONResponse(
            status_code=200,
            content={"status": "ok" if healthy else "degraded", "checks": checks},
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest, http_request: Request) -> QueryResponse:
        state: AppState = app.state.deps
        settings_ = state.settings
        started = time.perf_counter()

        if state.llm is None:
            raise HTTPException(status_code=503, detail="no LLM client is configured")

        try:
            chunk_set_id = request.chunk_set_id or state.resolve_chunk_set()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        config = RetrievalConfig(
            retriever=request.retriever or settings_.retriever,
            chunk_set_id=chunk_set_id,
            top_k=request.top_k or settings_.top_k,
            candidate_k=settings_.candidate_k,
            rrf_k=settings_.rrf_k,
            rerank=settings_.rerank_enabled if request.rerank is None else request.rerank,
            rerank_candidates=settings_.rerank_candidates,
            use_query_instruction=settings_.use_query_instruction,
            exact=settings_.exact_search,
        )

        # Built at startup, never here: loading an ONNX session inside a
        # request would put seconds of model load on a user's first query.
        reranker_ = state.reranker
        if config.rerank and reranker_ is None:
            raise HTTPException(
                status_code=503,
                detail="reranking was requested but no reranker is loaded",
            )

        try:
            with state.store_provider() as store:
                outcome = answer_question(
                    request.question,
                    settings=settings_,
                    store=store,
                    llm=state.llm,
                    config=config,
                    embedder=state.embedder,
                    reranker=reranker_,
                )
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (psycopg.Error, OSError) as exc:
            logger.exception("retrieval backend unavailable")
            raise HTTPException(
                status_code=503, detail=f"retrieval backend unavailable: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "query",
            extra={
                "request_id": getattr(http_request.state, "request_id", None),
                "retriever": config.retriever,
                "chunk_set_id": chunk_set_id,
                "abstained": outcome.abstained,
                "abstain_reason": outcome.abstain_reason,
                "citations": len(outcome.citations),
                "retrieved": len(outcome.retrieved),
                "llm_calls": outcome.llm_calls,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "cost_usd": outcome.cost_usd,
                "latency_ms": round(latency_ms, 2),
            },
        )

        return QueryResponse(
            answer=outcome.answer,
            abstained=outcome.abstained,
            citations=outcome.citations,
            retrieved_chunk_ids=outcome.retrieved_chunk_ids,
            latency_ms=round(latency_ms, 2),
            usage=outcome.usage,
            cost_usd=outcome.cost_usd,
            model=outcome.model,
            retriever=config.retriever,
        )

    return app
