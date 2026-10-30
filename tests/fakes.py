"""Test doubles shared by the generation and API tests.

Nothing here touches the network, a database, or a model file.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rbi_rag.generate.client import GenerationError, LLMResult
from rbi_rag.index.embed import FloatArray
from rbi_rag.index.store import InMemoryChunkStore
from rbi_rag.schemas import AnswerPayload, Chunk, TokenUsage

CHUNK_SET = 1

CAPITAL_TEXT = (
    "Every bank shall maintain a minimum Capital to Risk-weighted Assets Ratio "
    "of nine per cent on an ongoing basis."
)
TIER_ONE_TEXT = (
    "Tier 1 capital shall not be less than seven per cent of risk-weighted assets "
    "at any time during the financial year."
)
REPORTING_TEXT = (
    "Every bank shall report its capital position to the Reserve Bank within "
    "fifteen days of the close of each quarter."
)


def chunk(chunk_id: str, text: str, *, page: int = 3) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="11322",
        text=text,
        page_start=page,
        page_end=page,
        circular_number="RBI/DOR/2023-24/106",
        url="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/135MD.PDF",
        strategy="section_aware",
        target_tokens=512,
        token_count=24,
        section_title="3. Capital Requirements",
    )


def build_store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    store.add(
        CHUNK_SET,
        [
            chunk("c-capital", CAPITAL_TEXT),
            chunk("c-tier1", TIER_ONE_TEXT, page=4),
            chunk("c-reporting", REPORTING_TEXT, page=5),
        ],
        np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
    )
    return store


class FakeEmbedder:
    """Maps any query to a fixed vector, so dense ranking is predictable."""

    def __init__(self, vector: Sequence[float] = (1.0, 0.0)) -> None:
        self._vector = list(vector)

    @property
    def dimension(self) -> int:
        return len(self._vector)

    @property
    def max_content_tokens(self) -> int:
        return 510

    def embed_passages(self, texts: Sequence[str]) -> FloatArray:
        return np.array([self._vector for _ in texts], dtype=np.float32)

    def embed_query(self, text: str) -> FloatArray:
        return np.array([self._vector], dtype=np.float32)


class FakeReranker:
    """Returns a fixed score for every passage."""

    def __init__(self, score: float = 5.0) -> None:
        self.score_value = score

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [self.score_value for _ in passages]


class FakeLLM:
    """Replays scripted payloads, one per call.

    A scripted entry may be an exception, so failure paths are testable too.
    """

    def __init__(
        self,
        payloads: Sequence[AnswerPayload | Exception],
        *,
        model: str = "claude-sonnet-5-5",
        usage: TokenUsage | None = None,
    ) -> None:
        self._payloads = list(payloads)
        self.model = model
        self._usage = usage or TokenUsage(input_tokens=1000, output_tokens=200)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> LLMResult:
        self.calls.append((system, [dict(m) for m in messages]))
        if not self._payloads:
            raise GenerationError("FakeLLM ran out of scripted payloads")
        nxt = self._payloads.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResult(payload=nxt, usage=self._usage, model=self.model, stop_reason="end_turn")
