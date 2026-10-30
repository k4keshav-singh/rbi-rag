"""The review loop: the only path that can mark an item verified.

Nothing is auto-approved. Each candidate is shown with the source text behind
its gold span, and the owner answers. The script records that answer and
nothing else -- it has no notion of a "probably fine" item, because the value
of the verified flag comes entirely from a person having looked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rbi_rag.schemas import Chunk, EvalItem

Prompt = Callable[[str], str]

MENU = """\
  [k] keep        - verified, use it in reported metrics
  [e] edit        - change the question text, then keep
  [r] reject      - delete this candidate
  [s] skip        - leave unverified, decide later
  [q] quit        - save and stop
"""


@dataclass
class ReviewOutcome:
    """What the session did."""

    kept: int = 0
    rejected: int = 0
    skipped: int = 0
    edited: int = 0
    stopped_early: bool = False

    def describe(self) -> str:
        return (
            f"verified {self.kept} ({self.edited} after editing), "
            f"rejected {self.rejected}, left {self.skipped} unverified"
            + (" (stopped early)" if self.stopped_early else "")
        )


def render_item(item: EvalItem, sources: Sequence[Chunk]) -> str:
    """Show the candidate together with the text it claims to be answerable from."""
    lines = [
        "=" * 78,
        f"id            : {item.id}",
        f"type          : {item.question_type}",
        f"drafted by    : {item.drafted_by or 'hand-written'}",
        "",
        f"QUESTION      : {item.question}",
        "",
        f"expected      : {item.gold_answer_notes or '(none given)'}",
    ]

    if item.gold_spans:
        lines.append("")
        lines.append("GOLD SPANS")
        for span in item.gold_spans:
            pages = (
                f"p{span.page_start}"
                if span.page_start == span.page_end
                else f"pp{span.page_start}-{span.page_end}"
            )
            lines.append(f"  {span.doc_id} {pages}")
            if span.quote:
                lines.append(f"    quote: {span.quote}")
    else:
        lines.append("")
        lines.append("GOLD SPANS    : none - this is an unanswerable question.")
        lines.append("                Check it really is unanswerable from the corpus.")

    if sources:
        lines.append("")
        lines.append("SOURCE TEXT")
        for chunk in sources[:3]:
            excerpt = " ".join(chunk.text.split())
            lines.append(f"  [{chunk.chunk_id}]")
            lines.append(f"    {excerpt[:600]}")
    lines.append("")
    return "\n".join(lines)


def review_items(
    items: Sequence[EvalItem],
    source_lookup: Callable[[EvalItem], Sequence[Chunk]],
    prompt: Prompt,
    output: Callable[[str], None] = print,
) -> tuple[list[EvalItem], ReviewOutcome]:
    """Walk the unverified candidates. Returns the full updated set.

    ``prompt`` and ``output`` are injected so the loop is testable without a
    terminal.
    """
    outcome = ReviewOutcome()
    result: list[EvalItem] = []
    stopped = False

    for item in items:
        if item.verified_by_human or stopped:
            result.append(item)
            continue

        output(render_item(item, source_lookup(item)))
        output(MENU)
        choice = prompt("choice [k/e/r/s/q]: ").strip().lower()

        if choice == "k":
            result.append(item.model_copy(update={"verified_by_human": True}))
            outcome.kept += 1
        elif choice == "e":
            replacement = prompt("new question text: ").strip()
            if not replacement:
                output("empty question; leaving this one unverified.\n")
                result.append(item)
                outcome.skipped += 1
                continue
            result.append(
                item.model_copy(update={"question": replacement, "verified_by_human": True})
            )
            outcome.kept += 1
            outcome.edited += 1
        elif choice == "r":
            outcome.rejected += 1
            continue
        elif choice == "q":
            stopped = True
            outcome.stopped_early = True
            result.append(item)
        else:
            result.append(item)
            outcome.skipped += 1

    return result, outcome


def label_faithfulness(
    cases: Sequence[tuple[str, str, str]],
    prompt: Prompt,
    output: Callable[[str], None] = print,
) -> dict[str, bool]:
    """Collect the owner's faithfulness labels, for calibrating the judge.

    ``cases`` is (item_id, question, answer). Only explicit answers are
    recorded; anything skipped is simply absent, never assumed.
    """
    labels: dict[str, bool] = {}
    for item_id, question, answer in cases:
        output("=" * 78)
        output(f"id       : {item_id}")
        output(f"question : {question}")
        output("")
        output(f"answer   : {answer}")
        output("")
        output("Is every claim in this answer supported by the circulars it cited?")
        choice = prompt("  [y] yes  [n] no  [s] skip: ").strip().lower()
        if choice == "y":
            labels[item_id] = True
        elif choice == "n":
            labels[item_id] = False
    return labels
