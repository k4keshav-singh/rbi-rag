"""Per-request cost accounting.

Prices are reference data about the world, not configuration, so they live
here with the date they were taken. They change; a figure computed from a stale
table is wrong in a way that does not announce itself, which is why
``PRICE_TABLE_DATE`` is carried into the response.

A model with no entry yields ``None`` rather than a guessed figure. Rule 1 of
this project is that no number appears unless something measured it, and an
invented price would propagate straight into the Phase 4 cost column.
"""

from __future__ import annotations

from dataclasses import dataclass

from rbi_rag.schemas import TokenUsage

#: When these prices were taken from Anthropic's published pricing.
PRICE_TABLE_DATE = "2026-10-06"

#: Cache writes cost about 1.25x the base input rate at the default 5m TTL.
_CACHE_WRITE_MULTIPLIER = 1.25

#: Cache reads cost about 0.1x the base input rate unless stated otherwise.
_CACHE_READ_MULTIPLIER = 0.1


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    #: Explicit where Anthropic publishes one; derived from input otherwise.
    cache_read_per_mtok: float | None = None

    @property
    def cache_read(self) -> float:
        if self.cache_read_per_mtok is not None:
            return self.cache_read_per_mtok
        return self.input_per_mtok * _CACHE_READ_MULTIPLIER

    @property
    def cache_write(self) -> float:
        return self.input_per_mtok * _CACHE_WRITE_MULTIPLIER


PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.00, 50.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, cache_read_per_mtok=0.20),
    "claude-opus-5": ModelPrice(5.00, 25.00, cache_read_per_mtok=0.20),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, cache_read_per_mtok=0.20),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

_PER_MILLION = 1_000_000


def price_for(model: str) -> ModelPrice | None:
    """The price entry for a model, or None if we do not have one."""
    return PRICES.get(model)


def cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Cost of one call, or None when the model is not in the price table.

    Cached reads and cache writes are charged at their own rates rather than
    being folded into input tokens, since the difference is an order of
    magnitude and would otherwise quietly overstate the bill.
    """
    price = price_for(model)
    if price is None:
        return None
    total = (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    )
    return total / _PER_MILLION
