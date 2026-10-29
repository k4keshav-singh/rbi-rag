"""Token counting for the chunker.

Chunk size is the independent variable of the ablation, so it has to be counted
with the tokenizer the embedding model actually uses. Counting sits behind a
Protocol so that tests can inject a deterministic counter and stay offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a string."""

    def count(self, text: str) -> int:
        """Number of tokens, excluding any special/sentinel tokens."""
        ...


class WhitespaceTokenCounter:
    """Counts whitespace-separated words.

    For tests and for a rough offline estimate. Never use it to produce numbers
    that get reported: it is not the model's tokenizer and will disagree with it.
    """

    def count(self, text: str) -> int:
        return len(text.split())


class HuggingFaceTokenCounter:
    """Counts with the embedding model's own tokenizer.

    The tokenizer is downloaded once and then cached by the ``tokenizers``
    library, so constructing this needs network access the first time. Counting
    itself never does.
    """

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def count(self, text: str) -> int:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return len(encoding.ids)
