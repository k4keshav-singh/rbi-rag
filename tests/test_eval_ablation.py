"""The ablation: grid construction, evaluation, and which arms get generation."""

import pytest

from rbi_rag.eval.ablation import (
    RETRIEVERS,
    build_grid,
    evaluate_generation,
    evaluate_retrieval,
    select_generation_arms,
)
from rbi_rag.eval.report import ArmResult
from rbi_rag.generate.pipeline import AnswerOutcome
from rbi_rag.schemas import Chunk, EvalItem, GoldSpan, RetrievedChunk, TokenUsage

TEXT = "Every bank shall maintain a minimum ratio of nine per cent."


def hit(rank: int, *, page: int = 3, doc: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c{rank}",
            doc_id=doc,
            text=TEXT,
            page_start=page,
            page_end=page,
            url="https://example.test/x.PDF",
            strategy="fixed_tokens",
            target_tokens=512,
        ),
        score=1.0 / rank,
        rank=rank,
    )


def answerable(item_id: str = "f0001") -> EvalItem:
    return EvalItem(
        id=item_id,
        question="what ratio?",
        question_type="factual",
        gold_spans=[GoldSpan(doc_id="d1", page_start=3, page_end=3)],
        verified_by_human=True,
    )


def unanswerable(item_id: str = "u0001") -> EvalItem:
    return EvalItem(
        id=item_id,
        question="what colour is the sky?",
        question_type="unanswerable",
        gold_spans=[],
        verified_by_human=True,
    )


class TestBuildGrid:
    def test_every_combination_is_covered(self) -> None:
        arms = build_grid({("fixed_tokens", 256): 1, ("section_aware", 512): 2})
        # 2 chunk sets x 3 retrievers x 2 rerank settings
        assert len(arms) == 12
        assert len({a.arm_id for a in arms}) == 12

    def test_a_chunk_set_that_does_not_exist_produces_no_arms(self) -> None:
        assert build_grid({}) == []

    def test_arm_ids_describe_their_configuration(self) -> None:
        arms = build_grid({("section_aware", 512): 7})
        assert "section_aware-512-hybrid-rerank" in {a.arm_id for a in arms}

    def test_the_full_project_grid_is_thirty_six(self) -> None:
        chunk_sets = {
            (strategy, size): index
            for index, (strategy, size) in enumerate(
                [(s, n) for s in ("fixed_tokens", "section_aware") for n in (256, 512, 1024)],
                start=1,
            )
        }
        assert len(build_grid(chunk_sets)) == 36


class TestEvaluateRetrieval:
    def test_scores_a_perfect_retriever(self) -> None:
        metrics = evaluate_retrieval([answerable()], lambda q: [hit(1)], [1, 5])
        assert metrics["recall@1"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)

    def test_scores_a_retriever_that_misses(self) -> None:
        metrics = evaluate_retrieval([answerable()], lambda q: [hit(1, page=9)], [5])
        assert metrics["recall@5"] == pytest.approx(0.0)
        assert metrics["mrr"] == pytest.approx(0.0)

    def test_unanswerable_items_are_excluded(self) -> None:
        """They have nothing to recall; including them would distort the mean."""
        metrics = evaluate_retrieval([answerable(), unanswerable()], lambda q: [hit(1)], [5])
        assert metrics["recall@5"] == pytest.approx(1.0)

    def test_no_answerable_items_yields_nothing_rather_than_zero(self) -> None:
        metrics = evaluate_retrieval([unanswerable()], lambda q: [hit(1)], [5])
        assert metrics["recall@5"] is None
        assert metrics["mrr"] is None

    def test_averages_across_items(self) -> None:
        def retrieve(question: str) -> list[RetrievedChunk]:
            return [hit(1)] if question == "a" else [hit(1, page=9)]

        items = [
            answerable("f1").model_copy(update={"question": "a"}),
            answerable("f2").model_copy(update={"question": "b"}),
        ]
        assert evaluate_retrieval(items, retrieve, [5])["recall@5"] == pytest.approx(0.5)


def outcome(**kwargs: object) -> AnswerOutcome:
    base: dict[str, object] = {
        "answer": "Nine per cent.",
        "abstained": False,
        "usage": TokenUsage(input_tokens=100, output_tokens=10),
        "cost_usd": 0.001,
        "model": "claude-sonnet-5-5",
        "claimed_citations": 2,
        "valid_citations": 2,
    }
    return AnswerOutcome(**{**base, **kwargs})  # type: ignore[arg-type]


class TestEvaluateGeneration:
    def test_citation_precision_is_averaged(self) -> None:
        metrics = evaluate_generation(
            [answerable("f1"), answerable("f2")],
            lambda item: outcome(claimed_citations=4, valid_citations=2),
        )
        assert metrics["citation_precision"] == pytest.approx(0.5)

    def test_abstention_is_scored_against_question_type(self) -> None:
        def answer(item: EvalItem) -> AnswerOutcome:
            return outcome(abstained=not item.is_answerable, claimed_citations=0, valid_citations=0)

        metrics = evaluate_generation([answerable(), unanswerable()], answer)
        assert metrics["abstention_recall"] == pytest.approx(1.0)
        assert metrics["false_abstention_rate"] == pytest.approx(0.0)

    def test_faithfulness_comes_from_the_judge(self) -> None:
        metrics = evaluate_generation(
            [answerable("f1"), answerable("f2")],
            lambda item: outcome(),
            judge_faithful=lambda item, out: item.id == "f1",
        )
        assert metrics["faithfulness"] == pytest.approx(0.5)
        assert metrics["judged_answers"] == pytest.approx(2.0)

    def test_abstentions_are_not_judged(self) -> None:
        """There is no answer to check for support."""
        judged: list[str] = []

        def judge(item: EvalItem, out: AnswerOutcome) -> bool:
            judged.append(item.id)
            return True

        evaluate_generation([answerable()], lambda i: outcome(abstained=True), judge)
        assert judged == []

    def test_without_a_judge_faithfulness_is_not_invented(self) -> None:
        metrics = evaluate_generation([answerable()], lambda i: outcome())
        assert metrics["faithfulness"] is None

    def test_cost_is_summed(self) -> None:
        metrics = evaluate_generation(
            [answerable("f1"), answerable("f2")], lambda i: outcome(cost_usd=0.01)
        )
        assert metrics["cost_usd"] == pytest.approx(0.02)

    def test_an_unpriced_model_makes_total_cost_unknown_not_wrong(self) -> None:
        metrics = evaluate_generation([answerable()], lambda i: outcome(cost_usd=None))
        assert metrics["cost_usd"] is None

    def test_no_items(self) -> None:
        assert evaluate_generation([], lambda i: outcome()) == {}


def scored(arm: str, recall: float | None) -> ArmResult:
    strategy, size, retriever, rerank = arm.split("-")
    result = ArmResult(strategy, int(size), retriever, rerank == "rerank", chunk_set_id=1)
    result.metrics = {"recall@5": recall}
    return result


class TestSelectGenerationArms:
    def _results(self) -> list[ArmResult]:
        return [
            scored("section_aware-512-hybrid-rerank", 0.90),
            scored("section_aware-512-hybrid-norerank", 0.80),
            scored("section_aware-512-dense-rerank", 0.70),
            scored("section_aware-512-fts-rerank", 0.40),
            scored("fixed_tokens-256-hybrid-rerank", 0.60),
            scored("fixed_tokens-1024-dense-norerank", 0.30),
        ]

    def test_the_best_arm_is_always_included(self) -> None:
        chosen = select_generation_arms(self._results(), 6)
        assert chosen[0] == "section_aware-512-hybrid-rerank"

    def test_the_retriever_contrast_is_included(self) -> None:
        """The point of the subset is to answer a question, not to sample."""
        chosen = select_generation_arms(self._results(), 6)
        at_best_chunking = [c for c in chosen if c.startswith("section_aware-512-")]
        assert {c.split("-")[2] for c in at_best_chunking} >= set(RETRIEVERS)

    def test_the_rerank_toggle_is_included(self) -> None:
        chosen = select_generation_arms(self._results(), 6)
        assert "section_aware-512-hybrid-norerank" in chosen

    def test_the_limit_is_respected(self) -> None:
        assert len(select_generation_arms(self._results(), 3)) == 3

    def test_selection_is_deterministic(self) -> None:
        first = select_generation_arms(self._results(), 4)
        second = select_generation_arms(list(reversed(self._results())), 4)
        assert first == second

    def test_unmeasured_arms_fall_back_without_crashing(self) -> None:
        results = [scored("section_aware-512-hybrid-rerank", None)]
        assert select_generation_arms(results, 2) == ["section_aware-512-hybrid-rerank"]

    def test_a_zero_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            select_generation_arms(self._results(), 0)
