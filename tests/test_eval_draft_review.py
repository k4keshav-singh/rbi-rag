"""Question drafting and the review loop.

Between them these two modules decide what counts as ground truth, so the
tests are mostly about what they refuse to do: emit a question whose gold quote
is not really in the document, and verify anything without being told to.
"""

from collections.abc import Sequence

from rbi_rag.eval.draft import DraftedQuestion, DraftedUnanswerable, draft_from_chunks
from rbi_rag.eval.review import label_faithfulness, render_item, review_items
from rbi_rag.schemas import Chunk, EvalItem, GoldSpan

TEXT = "Every bank shall maintain a minimum ratio of nine per cent on an ongoing basis."


def chunk(chunk_id: str = "c1", text: str = TEXT) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="11322",
        text=text,
        page_start=3,
        page_end=4,
        circular_number="RBI/DOR/2023-24/106",
        url="https://example.test/x.PDF",
        strategy="section_aware",
        target_tokens=512,
    )


class FakeDraftClient:
    """Returns scripted proposals."""

    def __init__(
        self,
        proposals: Sequence[DraftedQuestion],
        unanswerable: DraftedUnanswerable | None = None,
    ) -> None:
        self._proposals = list(proposals)
        self._unanswerable = unanswerable or DraftedUnanswerable(question="plausible but absent")
        self.model = "claude-sonnet-5-5"

    def draft(self, chunk: Chunk) -> DraftedQuestion:
        return self._proposals.pop(0)

    def draft_unanswerable(self, chunk: Chunk) -> DraftedUnanswerable:
        return self._unanswerable


def good_proposal() -> DraftedQuestion:
    return DraftedQuestion(
        usable=True,
        question="What minimum ratio must a bank maintain?",
        question_type="factual",
        supporting_quote="a minimum ratio of nine per cent",
        answer_notes="Nine per cent, on an ongoing basis.",
    )


class TestDrafting:
    def test_builds_an_item_with_a_gold_span(self) -> None:
        items, rejected = draft_from_chunks(
            [chunk()], FakeDraftClient([good_proposal()]), unanswerable_every=0
        )
        assert rejected == []
        span = items[0].gold_spans[0]
        assert span.doc_id == "11322"
        assert span.page_start == 3
        assert span.page_end == 4
        assert span.quote == "a minimum ratio of nine per cent"

    def test_everything_drafted_is_unverified(self) -> None:
        """Otherwise the system would be graded on questions it wrote itself."""
        items, _ = draft_from_chunks(
            [chunk()], FakeDraftClient([good_proposal()]), unanswerable_every=0
        )
        assert all(item.verified_by_human is False for item in items)

    def test_records_which_model_proposed_it(self) -> None:
        items, _ = draft_from_chunks(
            [chunk()], FakeDraftClient([good_proposal()]), unanswerable_every=0
        )
        assert items[0].drafted_by == "claude-sonnet-5-5"
        assert items[0].drafted_at is not None

    def test_a_quote_that_is_not_in_the_excerpt_is_rejected(self) -> None:
        """A bad gold span would look like a retrieval failure forever after."""
        bad = good_proposal().model_copy(
            update={"supporting_quote": "a minimum ratio of twelve per cent"}
        )
        items, rejected = draft_from_chunks([chunk()], FakeDraftClient([bad]), unanswerable_every=0)
        assert items == []
        assert "not verbatim" in rejected[0].reason

    def test_line_wrapping_in_a_quote_is_tolerated(self) -> None:
        wrapped = chunk(text="Every bank shall maintain\na minimum ratio of nine per cent.")
        items, rejected = draft_from_chunks(
            [wrapped], FakeDraftClient([good_proposal()]), unanswerable_every=0
        )
        assert rejected == []
        assert len(items) == 1

    def test_an_unusable_excerpt_is_reported_not_silently_dropped(self) -> None:
        unusable = DraftedQuestion(usable=False)
        items, rejected = draft_from_chunks(
            [chunk()], FakeDraftClient([unusable]), unanswerable_every=0
        )
        assert items == []
        assert len(rejected) == 1

    def test_unanswerable_questions_are_interleaved(self) -> None:
        client = FakeDraftClient([good_proposal(), good_proposal()])
        items, _ = draft_from_chunks([chunk(), chunk()], client, unanswerable_every=2)
        types = [item.question_type for item in items]
        assert "unanswerable" in types

    def test_unanswerable_items_carry_no_gold_spans(self) -> None:
        client = FakeDraftClient([good_proposal()])
        items, _ = draft_from_chunks([chunk()], client, unanswerable_every=1)
        unanswerable = [i for i in items if i.question_type == "unanswerable"]
        assert unanswerable[0].gold_spans == []

    def test_ids_do_not_collide_with_existing_ones(self) -> None:
        client = FakeDraftClient([good_proposal()])
        items, _ = draft_from_chunks(
            [chunk()], client, existing_ids=["f0001"], unanswerable_every=0
        )
        assert items[0].id != "f0001"


def unverified(item_id: str = "f0001") -> EvalItem:
    return EvalItem(
        id=item_id,
        question="What minimum ratio must a bank maintain?",
        question_type="factual",
        gold_spans=[GoldSpan(doc_id="11322", page_start=3, page_end=4, quote="nine per cent")],
        gold_answer_notes="Nine per cent.",
        verified_by_human=False,
    )


class Replies:
    """Replays scripted keystrokes for the review loop."""

    def __init__(self, answers: Sequence[str]) -> None:
        self._answers = list(answers)

    def __call__(self, prompt: str) -> str:
        return self._answers.pop(0) if self._answers else "q"


def run_review(items: Sequence[EvalItem], answers: Sequence[str]):
    return review_items(items, lambda item: [chunk()], Replies(answers), output=lambda _: None)


class TestReview:
    def test_keep_marks_an_item_verified(self) -> None:
        updated, outcome = run_review([unverified()], ["k"])
        assert updated[0].verified_by_human is True
        assert outcome.kept == 1

    def test_reject_removes_the_item(self) -> None:
        updated, outcome = run_review([unverified()], ["r"])
        assert updated == []
        assert outcome.rejected == 1

    def test_skip_leaves_it_unverified(self) -> None:
        updated, outcome = run_review([unverified()], ["s"])
        assert updated[0].verified_by_human is False
        assert outcome.skipped == 1

    def test_an_unrecognised_key_never_verifies(self) -> None:
        """Verification has to be deliberate, not the result of a stray key."""
        updated, _ = run_review([unverified()], ["x"])
        assert updated[0].verified_by_human is False

    def test_edit_replaces_the_question_and_verifies(self) -> None:
        updated, outcome = run_review([unverified()], ["e", "A better question?"])
        assert updated[0].question == "A better question?"
        assert updated[0].verified_by_human is True
        assert outcome.edited == 1

    def test_an_empty_edit_does_not_verify(self) -> None:
        updated, _ = run_review([unverified()], ["e", "   "])
        assert updated[0].verified_by_human is False

    def test_quit_leaves_the_rest_untouched(self) -> None:
        updated, outcome = run_review([unverified("f1"), unverified("f2")], ["q"])
        assert all(item.verified_by_human is False for item in updated)
        assert outcome.stopped_early

    def test_already_verified_items_are_not_re_asked(self) -> None:
        verified = unverified().model_copy(update={"verified_by_human": True})
        updated, outcome = run_review([verified], [])
        assert updated[0].verified_by_human is True
        assert outcome.kept == 0

    def test_the_display_shows_the_source_text(self) -> None:
        rendered = render_item(unverified(), [chunk()])
        assert "nine per cent" in rendered
        assert "11322" in rendered

    def test_an_unanswerable_item_is_flagged_for_extra_care(self) -> None:
        item = unverified().model_copy(update={"question_type": "unanswerable", "gold_spans": []})
        assert "unanswerable" in render_item(item, [])


class TestFaithfulnessLabelling:
    def test_records_explicit_answers_only(self) -> None:
        labels = label_faithfulness(
            [("f1", "q1", "a1"), ("f2", "q2", "a2"), ("f3", "q3", "a3")],
            Replies(["y", "n", "s"]),
            output=lambda _: None,
        )
        assert labels == {"f1": True, "f2": False}

    def test_nothing_is_assumed_when_skipped(self) -> None:
        labels = label_faithfulness([("f1", "q", "a")], Replies(["s"]), output=lambda _: None)
        assert labels == {}
