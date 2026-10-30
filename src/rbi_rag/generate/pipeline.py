"""Answering: retrieve, gate, generate, validate, decide.

The pipeline fails closed. Every path that cannot produce a verified citation
ends in abstention rather than in an answer the reader cannot check, including
the one where the model writes something fluent and well-formed that simply is
not in the circulars.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from rbi_rag.config import Settings
from rbi_rag.generate.client import GenerationError, LLMClient
from rbi_rag.generate.cost import cost_usd
from rbi_rag.generate.prompt import SYSTEM_PROMPT, build_retry_message, build_user_message
from rbi_rag.generate.validate import validate_citations
from rbi_rag.index.embed import Embedder
from rbi_rag.index.store import ChunkStore
from rbi_rag.retrieve.engine import retrieve
from rbi_rag.retrieve.rerank import Reranker
from rbi_rag.schemas import Citation, RetrievalConfig, RetrievedChunk, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class AnswerOutcome:
    """Everything POST /query needs, plus why it came out that way."""

    answer: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    model: str = ""
    abstain_reason: str | None = None
    llm_calls: int = 0

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [item.chunk.chunk_id for item in self.retrieved]


def gate_score(chunks: Sequence[RetrievedChunk]) -> float | None:
    """A top-result confidence score that is comparable across retrievers.

    The reranker score where there is one, otherwise the raw dense cosine.
    Returns None when neither exists -- under a keyword-only retriever there is
    no comparable number, and ``ts_rank_cd`` is not one. The alternative,
    thresholding an RRF score, compares roughly 0.03 against a cosine of 0.83
    and is meaningless.
    """
    if not chunks:
        return None
    top = chunks[0]
    if top.rerank_score is not None:
        return top.rerank_score
    return top.dense_score


def answer_question(
    question: str,
    *,
    settings: Settings,
    store: ChunkStore,
    llm: LLMClient,
    config: RetrievalConfig,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> AnswerOutcome:
    """Answer one question, or decline to."""
    retrieved = retrieve(question, store=store, config=config, embedder=embedder, reranker=reranker)

    if not retrieved:
        return _abstain(settings, retrieved, "nothing was retrieved for this question")

    threshold = settings.abstain_min_score
    if threshold is not None:
        score = gate_score(retrieved)
        if score is not None and score < threshold:
            return _abstain(
                settings,
                retrieved,
                f"top retrieval score {score:.4f} is below the threshold {threshold}",
            )

    system = SYSTEM_PROMPT
    messages: list[dict[str, str]] = [
        {"role": "user", "content": build_user_message(question, retrieved)}
    ]

    usage = TokenUsage()
    calls = 0
    model = ""
    last_violations: list[str] = []

    for attempt in range(settings.citation_retry_attempts + 1):
        try:
            result = llm.complete(system, messages)
        except GenerationError:
            logger.exception("generation failed", extra={"attempt": attempt})
            raise

        calls += 1
        model = result.model
        usage = _add(usage, result.usage)
        payload = result.payload

        if not payload.sufficient_evidence:
            return _abstain(
                settings,
                retrieved,
                "the model judged the excerpts insufficient",
                usage=usage,
                model=model,
                calls=calls,
            )

        validation = validate_citations(payload.citations, retrieved)

        if validation.ok and validation.citations:
            return AnswerOutcome(
                answer=payload.answer.strip(),
                abstained=False,
                citations=validation.citations,
                retrieved=list(retrieved),
                usage=usage,
                cost_usd=cost_usd(model, usage),
                model=model,
                llm_calls=calls,
            )

        last_violations = list(validation.violations)
        if validation.ok and not validation.citations:
            # Well-formed, confident, and unverifiable. Treated as a failure,
            # because an uncited answer is exactly what this system must not
            # emit however plausible it reads.
            last_violations = ["the answer cited nothing that could be verified"]

        logger.warning(
            "citation validation failed",
            extra={"attempt": attempt, "violations": last_violations},
        )

        if attempt < settings.citation_retry_attempts:
            messages = [
                *messages,
                {"role": "assistant", "content": payload.answer},
                {"role": "user", "content": build_retry_message(last_violations)},
            ]

    return _abstain(
        settings,
        retrieved,
        f"citations could not be verified after {calls} attempt(s): "
        f"{'; '.join(last_violations[:3])}",
        usage=usage,
        model=model,
        calls=calls,
    )


def _abstain(
    settings: Settings,
    retrieved: Sequence[RetrievedChunk],
    reason: str,
    *,
    usage: TokenUsage | None = None,
    model: str = "",
    calls: int = 0,
) -> AnswerOutcome:
    spent = usage or TokenUsage()
    logger.info("abstained", extra={"reason": reason})
    return AnswerOutcome(
        answer=settings.abstention_message,
        abstained=True,
        citations=[],
        retrieved=list(retrieved),
        usage=spent,
        cost_usd=cost_usd(model, spent) if model else None,
        model=model or settings.anthropic_model,
        abstain_reason=reason,
        llm_calls=calls,
    )


def _add(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Accumulate usage across retries, so cost covers the whole request."""
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
    )
