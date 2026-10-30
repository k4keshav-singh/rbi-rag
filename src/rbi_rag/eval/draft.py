"""Proposing candidate evaluation questions from indexed chunks.

Everything this module writes carries ``verified_by_human=false``. A drafted
question is a suggestion for the owner to accept, edit or throw away; it is not
ground truth, and treating it as such would mean grading the system against
questions the system's own model wrote.

The supporting quote is checked against the source chunk before an item is
emitted, for the same reason citations are checked in generation: a drafted
gold span whose quote is not actually in the document would silently make a
question unanswerable and show up later as a retrieval failure.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from rbi_rag.generate.validate import normalise
from rbi_rag.schemas import Chunk, EvalItem, GoldSpan, QuestionType

DRAFT_SYSTEM_PROMPT = """\
You are helping build an evaluation set for a question-answering system over \
RBI Master Directions.

Given one excerpt from a circular, write a question that the excerpt answers.

Requirements:
- The question must be answerable from this excerpt alone, and specific enough \
that a different circular would not answer it by accident.
- Write it the way someone with a real regulatory question would, not as a \
quiz about the text. Do not write "according to the excerpt" or refer to the \
excerpt at all.
- supporting_quote must be copied character for character from the excerpt. Do \
not paraphrase, tidy punctuation, or join separated passages.
- answer_notes states what a correct answer must contain, in one or two lines.

If the excerpt is boilerplate, a heading, a table of contents, or otherwise \
has no substantive content to ask about, set usable to false and leave the \
other fields empty."""

UNANSWERABLE_SYSTEM_PROMPT = """\
You are helping build an evaluation set for a question-answering system over \
RBI Master Directions.

Write a question that sounds like it belongs to this corpus but that the \
excerpt below does NOT answer, and that a reader could not answer from it.

The question must be plausible: same subject area, same register, the kind of \
thing a banker might genuinely ask. It must not be absurd or off-topic, \
because the point is to test whether the system declines when it lacks \
evidence, not whether it notices nonsense.

Do not reference the excerpt. Leave answer_notes explaining briefly why the \
corpus excerpt does not answer it."""


class DraftedQuestion(BaseModel):
    """What the drafting model returns for one chunk."""

    usable: bool = Field(description="False when the excerpt has nothing worth asking about.")
    question: str = ""
    question_type: Literal["factual", "multi_hop"] = "factual"
    supporting_quote: str = ""
    answer_notes: str = ""


class DraftedUnanswerable(BaseModel):
    """A plausible question the corpus does not answer."""

    question: str
    answer_notes: str = ""


class DraftClient(Protocol):
    """Proposes questions. Mocked in tests."""

    def draft(self, chunk: Chunk) -> DraftedQuestion: ...

    def draft_unanswerable(self, chunk: Chunk) -> DraftedUnanswerable: ...

    @property
    def model(self) -> str: ...


class DraftRejection(BaseModel):
    """Why a proposal did not become an item."""

    chunk_id: str
    reason: str


def _item_id(prefix: str, index: int) -> str:
    return f"{prefix}{index:04d}"


def draft_from_chunks(
    chunks: Sequence[Chunk],
    client: DraftClient,
    *,
    existing_ids: Sequence[str] = (),
    unanswerable_every: int = 5,
) -> tuple[list[EvalItem], list[DraftRejection]]:
    """Propose one question per chunk, plus unanswerable questions periodically.

    Returns the items and the rejections, so a thin yield is visible rather
    than looking like the corpus simply produced fewer questions.
    """
    taken = set(existing_ids)
    items: list[EvalItem] = []
    rejections: list[DraftRejection] = []
    counters = {"f": 0, "m": 0, "u": 0}
    now = dt.datetime.now(tz=dt.UTC)

    def allocate(question_type: QuestionType) -> str:
        prefix = {"factual": "f", "multi_hop": "m", "unanswerable": "u"}[question_type]
        while True:
            counters[prefix] += 1
            candidate = _item_id(prefix, counters[prefix])
            if candidate not in taken:
                taken.add(candidate)
                return candidate

    for position, chunk in enumerate(chunks):
        proposal = client.draft(chunk)
        if not proposal.usable or not proposal.question.strip():
            rejections.append(
                DraftRejection(chunk_id=chunk.chunk_id, reason="model marked excerpt unusable")
            )
        elif normalise(proposal.supporting_quote) not in normalise(chunk.text):
            # Same discipline as citation validation: an unverifiable quote is
            # not evidence, and a bad gold span poisons every later metric.
            rejections.append(
                DraftRejection(
                    chunk_id=chunk.chunk_id,
                    reason="supporting quote is not verbatim in the excerpt",
                )
            )
        else:
            question_type: QuestionType = proposal.question_type
            items.append(
                EvalItem(
                    id=allocate(question_type),
                    question=proposal.question.strip(),
                    question_type=question_type,
                    gold_spans=[
                        GoldSpan(
                            doc_id=chunk.doc_id,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            quote=proposal.supporting_quote.strip(),
                        )
                    ],
                    gold_answer_notes=proposal.answer_notes.strip(),
                    verified_by_human=False,
                    drafted_by=client.model,
                    drafted_at=now,
                    source_doc_id=chunk.doc_id,
                )
            )

        if unanswerable_every and (position + 1) % unanswerable_every == 0:
            unanswerable = client.draft_unanswerable(chunk)
            if unanswerable.question.strip():
                items.append(
                    EvalItem(
                        id=allocate("unanswerable"),
                        question=unanswerable.question.strip(),
                        question_type="unanswerable",
                        gold_spans=[],
                        gold_answer_notes=unanswerable.answer_notes.strip(),
                        verified_by_human=False,
                        drafted_by=client.model,
                        drafted_at=now,
                        source_doc_id=chunk.doc_id,
                    )
                )

    return items, rejections
