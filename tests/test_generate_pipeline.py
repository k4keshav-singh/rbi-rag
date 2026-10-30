"""The answering pipeline.

The pipeline must fail closed: every route that cannot produce a verified
citation has to end in abstention rather than in an unverifiable answer.
"""

import pytest

from rbi_rag.config import Settings
from rbi_rag.generate.client import GenerationError
from rbi_rag.generate.pipeline import answer_question, gate_score
from rbi_rag.schemas import AnswerPayload, CitationClaim, RetrievalConfig, RetrievedChunk
from tests.fakes import (
    CAPITAL_TEXT,
    CHUNK_SET,
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    build_store,
    chunk,
)

GOOD_QUOTE = "nine per cent on an ongoing basis"


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def config(**overrides: object) -> RetrievalConfig:
    base: dict[str, object] = {"chunk_set_id": CHUNK_SET, "retriever": "dense", "top_k": 3}
    return RetrievalConfig(**{**base, **overrides})  # type: ignore[arg-type]


def good_payload() -> AnswerPayload:
    return AnswerPayload(
        answer="Banks must maintain a CRAR of nine per cent.",
        citations=[CitationClaim(chunk_id="c-capital", quote=GOOD_QUOTE)],
        sufficient_evidence=True,
    )


def run(llm: FakeLLM, *, cfg: RetrievalConfig | None = None, **setting_overrides: object):
    return answer_question(
        "what capital ratio must banks maintain",
        settings=settings(**setting_overrides),
        store=build_store(),
        llm=llm,
        config=cfg or config(),
        embedder=FakeEmbedder(),
    )


class TestHappyPath:
    def test_returns_the_answer_with_citations(self) -> None:
        outcome = run(FakeLLM([good_payload()]))
        assert not outcome.abstained
        assert "nine per cent" in outcome.answer
        assert [c.chunk_id for c in outcome.citations] == ["c-capital"]

    def test_reports_what_was_retrieved(self) -> None:
        outcome = run(FakeLLM([good_payload()]))
        assert "c-capital" in outcome.retrieved_chunk_ids

    def test_reports_usage_and_cost(self) -> None:
        outcome = run(FakeLLM([good_payload()]))
        assert outcome.usage.input_tokens == 1000
        assert outcome.cost_usd is not None
        assert outcome.cost_usd > 0

    def test_calls_the_model_once(self) -> None:
        llm = FakeLLM([good_payload()])
        run(llm)
        assert len(llm.calls) == 1

    def test_the_prompt_contains_the_retrieved_text_and_ids(self) -> None:
        llm = FakeLLM([good_payload()])
        run(llm)
        _, messages = llm.calls[0]
        user_turn = messages[0]["content"]
        assert "c-capital" in user_turn
        assert CAPITAL_TEXT[:40] in user_turn


class TestAbstention:
    def test_abstains_when_nothing_is_retrieved(self) -> None:
        llm = FakeLLM([good_payload()])
        outcome = answer_question(
            "unrelated question",
            settings=settings(),
            store=build_store(),
            llm=llm,
            config=config(chunk_set_id=999),
            embedder=FakeEmbedder(),
        )
        assert outcome.abstained
        assert llm.calls == [], "must not pay for a generation with no context"

    def test_abstains_when_the_model_says_the_evidence_is_thin(self) -> None:
        payload = AnswerPayload(answer="", citations=[], sufficient_evidence=False)
        outcome = run(FakeLLM([payload]))
        assert outcome.abstained
        assert outcome.abstain_reason is not None
        assert "insufficient" in outcome.answer.lower()

    def test_abstains_when_the_answer_cites_nothing(self) -> None:
        """A confident uncited answer is exactly what must not be emitted."""
        payload = AnswerPayload(
            answer="Banks must maintain nine per cent.", citations=[], sufficient_evidence=True
        )
        outcome = run(FakeLLM([payload, payload]), citation_retry_attempts=1)
        assert outcome.abstained

    def test_the_retrieval_gate_can_skip_the_model_entirely(self) -> None:
        llm = FakeLLM([good_payload()])
        outcome = answer_question(
            "what capital ratio must banks maintain",
            settings=settings(abstain_min_score=0.99),
            store=build_store(),
            llm=llm,
            config=config(),
            # Far from every stored chunk, so the best cosine is ~0.71.
            embedder=FakeEmbedder(vector=(0.7071, -0.7071)),
        )
        assert outcome.abstained
        assert llm.calls == []

    def test_the_gate_is_off_by_default(self) -> None:
        """No threshold is invented before Phase 4 can tune one."""
        assert settings().abstain_min_score is None


class TestCitationRetry:
    def test_a_fabricated_quote_triggers_a_retry(self) -> None:
        bad = AnswerPayload(
            answer="Banks must maintain twelve per cent.",
            citations=[CitationClaim(chunk_id="c-capital", quote="twelve per cent at all times")],
            sufficient_evidence=True,
        )
        llm = FakeLLM([bad, good_payload()])
        outcome = run(llm, citation_retry_attempts=1)
        assert not outcome.abstained
        assert len(llm.calls) == 2

    def test_the_retry_turn_names_the_violation(self) -> None:
        bad = AnswerPayload(
            answer="...",
            citations=[CitationClaim(chunk_id="c-invented", quote="something made up here")],
            sufficient_evidence=True,
        )
        llm = FakeLLM([bad, good_payload()])
        run(llm, citation_retry_attempts=1)
        _, messages = llm.calls[1]
        assert "c-invented" in messages[-1]["content"]

    def test_abstains_when_the_retry_also_fabricates(self) -> None:
        bad = AnswerPayload(
            answer="Banks must maintain twelve per cent.",
            citations=[CitationClaim(chunk_id="c-capital", quote="twelve per cent at all times")],
            sufficient_evidence=True,
        )
        llm = FakeLLM([bad, bad])
        outcome = run(llm, citation_retry_attempts=1)
        assert outcome.abstained
        assert outcome.citations == []
        assert len(llm.calls) == 2

    def test_zero_retries_abstains_immediately(self) -> None:
        bad = AnswerPayload(
            answer="...",
            citations=[CitationClaim(chunk_id="c-invented", quote="something made up here")],
            sufficient_evidence=True,
        )
        llm = FakeLLM([bad])
        outcome = run(llm, citation_retry_attempts=0)
        assert outcome.abstained
        assert len(llm.calls) == 1

    def test_usage_accumulates_across_retries(self) -> None:
        """The request is billed for both calls, so both must be reported."""
        bad = AnswerPayload(
            answer="...",
            citations=[CitationClaim(chunk_id="c-invented", quote="something made up here")],
            sufficient_evidence=True,
        )
        llm = FakeLLM([bad, good_payload()])
        outcome = run(llm, citation_retry_attempts=1)
        assert outcome.usage.input_tokens == 2000
        assert outcome.llm_calls == 2


class TestFailures:
    def test_a_generation_error_propagates(self) -> None:
        with pytest.raises(GenerationError):
            run(FakeLLM([GenerationError("model declined")]))


class TestGateScore:
    def _hit(self, **kwargs: object) -> RetrievedChunk:
        base: dict[str, object] = {
            "chunk": chunk("c1", CAPITAL_TEXT),
            "score": 0.03,
            "rank": 1,
        }
        return RetrievedChunk(**{**base, **kwargs})  # type: ignore[arg-type]

    def test_prefers_the_reranker_score(self) -> None:
        assert gate_score([self._hit(rerank_score=4.2, dense_score=0.8)]) == 4.2

    def test_falls_back_to_dense_cosine(self) -> None:
        assert gate_score([self._hit(dense_score=0.83)]) == 0.83

    def test_returns_none_when_no_comparable_score_exists(self) -> None:
        """Under FTS alone there is no comparable number, and RRF is not one."""
        assert gate_score([self._hit()]) is None

    def test_empty_results_have_no_score(self) -> None:
        assert gate_score([]) is None

    def test_never_uses_the_rrf_score(self) -> None:
        """RRF tops out near 0.03; a cosine threshold against it is meaningless."""
        assert gate_score([self._hit(score=0.0328)]) != 0.0328


class TestReranking:
    def test_rerank_scores_reach_the_gate(self) -> None:
        outcome = answer_question(
            "what capital ratio must banks maintain",
            settings=settings(abstain_min_score=10.0),
            store=build_store(),
            llm=FakeLLM([good_payload()]),
            config=config(rerank=True),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(score=1.0),
        )
        assert outcome.abstained
        assert "1.0000" in (outcome.abstain_reason or "")
