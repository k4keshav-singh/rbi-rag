"""LLM-as-judge for faithfulness.

The judge is asked one narrow question: is every claim in the answer supported
by the excerpts the answer cited? Not whether the answer is correct, not
whether it is well written, not whether it agrees with the judge's own
knowledge of Indian banking regulation -- the last of which would quietly turn
a faithfulness metric into a knowledge test of the judge.

A judge is a measuring instrument, and an uncalibrated instrument produces
numbers about itself. `calibration.py` compares its verdicts with the owner's
on a subset, and the report prints that agreement next to the score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rbi_rag.config import Settings
from rbi_rag.schemas import Citation, TokenUsage

JUDGE_SYSTEM_PROMPT = """\
You are checking whether an answer is supported by the excerpts it cites.

You are given a question, an answer, and the excerpts the answer cited. Decide \
one thing only: is every factual claim in the answer supported by those \
excerpts?

Judge support, not correctness. If a claim is true of Indian banking regulation \
but is not stated in the excerpts, it is unsupported. If a claim is stated in \
the excerpts but you believe it is out of date, it is still supported. Your own \
knowledge of the subject is not evidence here.

Treat as unsupported: a figure, threshold, date or deadline that differs from \
the excerpt; a condition or exception the excerpt does not state; a \
generalisation beyond the excerpt's scope; and any claim attributed to an \
excerpt that does not contain it.

Do not treat as unsupported: rewording, summarising, or reordering content that \
is genuinely present, and ordinary connective phrasing that asserts nothing.

List each unsupported claim verbatim from the answer. An answer with no \
unsupported claims is faithful."""


class FaithfulnessVerdict(BaseModel):
    """One judge decision about one answer."""

    model_config = ConfigDict(frozen=True)

    faithful: bool = Field(description="True when every claim is supported by the excerpts.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Verbatim spans of the answer that are unsupported."
    )
    reasoning: str = Field(default="", description="One or two sentences; not scored.")


class JudgeResult(BaseModel):
    """A verdict plus what it cost to obtain."""

    model_config = ConfigDict(frozen=True)

    verdict: FaithfulnessVerdict
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""


class Judge(Protocol):
    """Scores one answer for faithfulness against its citations."""

    def judge(self, question: str, answer: str, citations: Sequence[Citation]) -> JudgeResult: ...


def render_cited_excerpts(citations: Sequence[Citation]) -> str:
    """The evidence the answer actually leaned on."""
    if not citations:
        return "(the answer cited nothing)"
    blocks: list[str] = []
    for index, citation in enumerate(citations, start=1):
        source = citation.circular_number or citation.chunk_id
        pages = (
            f"page {citation.page_start}"
            if citation.page_start == citation.page_end
            else f"pages {citation.page_start}-{citation.page_end}"
        )
        blocks.append(f"[{index}] {source} ({pages})\n{citation.quote or '(no quote given)'}")
    return "\n\n".join(blocks)


def build_judge_message(question: str, answer: str, citations: Sequence[Citation]) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Answer under review:\n{answer}\n\n"
        f"Excerpts the answer cited:\n{render_cited_excerpts(citations)}"
    )


class AnthropicJudge:
    """The real judge. Separate call and prompt from answer generation."""

    def __init__(self, settings: Settings, model: str | None = None) -> None:
        import anthropic

        if settings.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; the judge cannot run.")
        self._settings = settings
        self._anthropic = anthropic
        self.model = model or settings.judge_model or settings.anthropic_model
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.anthropic_timeout_seconds,
            max_retries=settings.anthropic_max_retries,
        )

    def judge(self, question: str, answer: str, citations: Sequence[Citation]) -> JudgeResult:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self._settings.anthropic_max_tokens,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_judge_message(question, answer, citations)}
            ],
            output_format=FaithfulnessVerdict,
            output_config={"effort": self._settings.anthropic_effort},
        )

        verdict: FaithfulnessVerdict | None = None
        for block in response.content:
            parsed = getattr(block, "parsed_output", None)
            if isinstance(parsed, FaithfulnessVerdict):
                verdict = parsed
                break
        if verdict is None:
            raise RuntimeError(
                f"judge returned no structured verdict (stop_reason={response.stop_reason})"
            )

        usage = response.usage
        return JudgeResult(
            verdict=verdict,
            usage=TokenUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            model=response.model,
        )
