"""The ablation: sweep configurations and measure each one.

Retrieval is swept across the whole grid because it costs nothing and is
deterministic. Generation is measured on a chosen subset, because answering and
judging every arm multiplies LLM spend by the size of the grid to learn
something the retrieval table has mostly already said. Which arms make the
subset is derived from the retrieval results rather than picked by hand, so the
choice is reproducible.

Everything here takes callables rather than a database handle, so the logic is
unit-tested with no Postgres, no model and no network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rbi_rag.eval.metrics import (
    abstention_rates,
    citation_precision,
    mean_ignoring_none,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rbi_rag.eval.report import ArmResult
from rbi_rag.generate.pipeline import AnswerOutcome
from rbi_rag.schemas import ChunkStrategy, EvalItem, RetrievedChunk, RetrieverName

#: Retrieval is swept over every combination of these.
RETRIEVERS: tuple[RetrieverName, ...] = ("dense", "fts", "hybrid")
STRATEGIES: tuple[ChunkStrategy, ...] = ("fixed_tokens", "section_aware")
RERANK_OPTIONS: tuple[bool, ...] = (False, True)

#: The metric the generation subset is chosen by.
PRIMARY_METRIC = "recall@5"

Retrieve = Callable[[str], Sequence[RetrievedChunk]]
Answer = Callable[[EvalItem], AnswerOutcome]
JudgeFaithful = Callable[[EvalItem, AnswerOutcome], bool | None]


@dataclass(frozen=True)
class ArmSpec:
    """One point in the grid."""

    strategy: ChunkStrategy
    target_tokens: int
    retriever: RetrieverName
    rerank: bool
    chunk_set_id: int

    @property
    def arm_id(self) -> str:
        return (
            f"{self.strategy}-{self.target_tokens}-{self.retriever}-"
            f"{'rerank' if self.rerank else 'norerank'}"
        )

    def to_result(self) -> ArmResult:
        return ArmResult(
            strategy=self.strategy,
            target_tokens=self.target_tokens,
            retriever=self.retriever,
            rerank=self.rerank,
            chunk_set_id=self.chunk_set_id,
        )


def build_grid(chunk_sets: dict[tuple[ChunkStrategy, int], int]) -> list[ArmSpec]:
    """Every arm, given the chunk sets that exist.

    ``chunk_sets`` maps (strategy, target_tokens) to a chunk_set_id, so a
    configuration that was never indexed simply produces no arms rather than
    failing halfway through a sweep.
    """
    arms: list[ArmSpec] = []
    for (strategy, size), chunk_set_id in sorted(chunk_sets.items()):
        for retriever in RETRIEVERS:
            for rerank in RERANK_OPTIONS:
                arms.append(
                    ArmSpec(
                        strategy=strategy,
                        target_tokens=size,
                        retriever=retriever,
                        rerank=rerank,
                        chunk_set_id=chunk_set_id,
                    )
                )
    return arms


# -- retrieval ---------------------------------------------------------------


def evaluate_retrieval(
    items: Sequence[EvalItem],
    retrieve: Retrieve,
    k_values: Sequence[int],
) -> dict[str, float | None]:
    """Recall@k, MRR and nDCG@k over gold spans.

    Unanswerable items are skipped: they have no spans to recall, and scoring
    them zero would penalise the system for correctly having nothing to find.
    """
    answerable = [item for item in items if item.is_answerable and item.gold_spans]
    if not answerable:
        empty: dict[str, float | None] = {f"recall@{k}": None for k in k_values}
        empty["mrr"] = None
        return empty

    per_item: dict[str, list[float | None]] = {f"recall@{k}": [] for k in k_values}
    per_item |= {f"ndcg@{k}": [] for k in k_values}
    per_item["mrr"] = []

    for item in answerable:
        retrieved = list(retrieve(item.question))
        for k in k_values:
            per_item[f"recall@{k}"].append(recall_at_k(retrieved, item.gold_spans, k))
            per_item[f"ndcg@{k}"].append(ndcg_at_k(retrieved, item.gold_spans, k))
        per_item["mrr"].append(reciprocal_rank(retrieved, item.gold_spans))

    return {name: mean_ignoring_none(values) for name, values in per_item.items()}


# -- generation --------------------------------------------------------------


def evaluate_generation(
    items: Sequence[EvalItem],
    answer: Answer,
    judge_faithful: JudgeFaithful | None = None,
) -> dict[str, float | None]:
    """Citation precision, abstention behaviour and faithfulness."""
    if not items:
        return {}

    precisions: list[float | None] = []
    abstention_inputs: list[tuple[bool, bool]] = []
    faithful_votes: list[bool] = []
    total_cost = 0.0
    cost_known = True

    for item in items:
        outcome = answer(item)
        abstention_inputs.append((item.is_answerable, outcome.abstained))
        precisions.append(citation_precision(outcome.valid_citations, outcome.claimed_citations))
        if outcome.cost_usd is None:
            cost_known = False
        else:
            total_cost += outcome.cost_usd

        if judge_faithful is not None and not outcome.abstained:
            verdict = judge_faithful(item, outcome)
            if verdict is not None:
                faithful_votes.append(verdict)

    metrics: dict[str, float | None] = {
        "citation_precision": mean_ignoring_none(precisions),
        "faithfulness": (sum(faithful_votes) / len(faithful_votes) if faithful_votes else None),
        "judged_answers": float(len(faithful_votes)),
        "cost_usd": total_cost if cost_known else None,
    }
    metrics.update(abstention_rates(abstention_inputs))
    return metrics


# -- choosing which arms get generation --------------------------------------


def select_generation_arms(
    results: Sequence[ArmResult],
    limit: int,
    primary_metric: str = PRIMARY_METRIC,
) -> list[str]:
    """Pick the arms worth paying to generate and judge.

    The best arm on the primary retrieval metric, then the contrasts that
    actually answer a question: the other retrievers at that same chunk
    configuration, and the rerank toggle. Padding, if any room is left, follows
    the retrieval ranking. Derived from measurements, so a re-run picks the
    same arms.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    scored = [r for r in results if r.metrics.get(primary_metric) is not None]
    if not scored:
        return [r.arm_id for r in results[:limit]]

    ranked = sorted(
        scored,
        key=lambda r: (-(r.metrics[primary_metric] or 0.0), r.arm_id),
    )
    best = ranked[0]
    chosen: list[str] = [best.arm_id]

    def add(candidate: ArmResult) -> None:
        if candidate.arm_id not in chosen and len(chosen) < limit:
            chosen.append(candidate.arm_id)

    same_chunking = [
        r for r in ranked if r.strategy == best.strategy and r.target_tokens == best.target_tokens
    ]
    # Other retrievers at the winning chunk configuration.
    for retriever in RETRIEVERS:
        for candidate in same_chunking:
            if candidate.retriever == retriever and candidate.rerank == best.rerank:
                add(candidate)
                break
    # The rerank toggle, so its contribution is isolated.
    for candidate in same_chunking:
        if candidate.retriever == best.retriever and candidate.rerank != best.rerank:
            add(candidate)
            break
    # Any remaining room goes to the next best arms overall.
    for candidate in ranked:
        add(candidate)

    return chosen
