"""Cost accounting.

Every figure here is hand-computed from the published per-million rates, so a
change to the price table that breaks the arithmetic shows up as a failure
rather than as a quietly wrong number in a results column.
"""

import pytest

from rbi_rag.generate.cost import PRICE_TABLE_DATE, PRICES, cost_usd, price_for
from rbi_rag.schemas import TokenUsage


class TestPriceTable:
    def test_carries_the_date_it_was_taken(self) -> None:
        """A stale price is wrong in a way that does not announce itself."""
        assert PRICE_TABLE_DATE

    def test_the_project_default_model_is_priced(self) -> None:
        assert price_for("claude-sonnet-5-5") is not None

    def test_an_unknown_model_has_no_price(self) -> None:
        assert price_for("claude-does-not-exist") is None

    def test_output_always_costs_more_than_input(self) -> None:
        for name, price in PRICES.items():
            assert price.output_per_mtok > price.input_per_mtok, name

    def test_cache_reads_are_much_cheaper_than_fresh_input(self) -> None:
        for name, price in PRICES.items():
            assert price.cache_read < price.input_per_mtok, name

    def test_cache_writes_cost_more_than_fresh_input(self) -> None:
        for name, price in PRICES.items():
            assert price.cache_write > price.input_per_mtok, name


class TestCostUsd:
    def test_sonnet_arithmetic(self) -> None:
        # 1,000,000 input at $2 + 1,000,000 output at $10 = $12 exactly.
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(12.0)

    def test_a_realistic_request(self) -> None:
        # 8,000 input at $2/M = $0.016; 500 output at $10/M = $0.005.
        usage = TokenUsage(input_tokens=8_000, output_tokens=500)
        assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(0.021)

    def test_cached_reads_are_billed_at_the_cache_rate(self) -> None:
        cached = TokenUsage(cache_read_tokens=1_000_000)
        fresh = TokenUsage(input_tokens=1_000_000)
        assert cost_usd("claude-sonnet-5-5", cached) == pytest.approx(0.20)
        assert cost_usd("claude-sonnet-5-5", fresh) == pytest.approx(2.00)

    def test_cache_writes_carry_their_premium(self) -> None:
        written = TokenUsage(cache_write_tokens=1_000_000)
        assert cost_usd("claude-sonnet-5-5", written) == pytest.approx(2.50)

    def test_opus_costs_twice_what_sonnet_does(self) -> None:
        usage = TokenUsage(input_tokens=10_000, output_tokens=1_000)
        sonnet = cost_usd("claude-sonnet-5-5", usage)
        opus = cost_usd("claude-opus-5-5", usage)
        assert sonnet is not None and opus is not None
        assert opus == pytest.approx(sonnet * 2)

    def test_zero_usage_costs_nothing(self) -> None:
        assert cost_usd("claude-sonnet-5-5", TokenUsage()) == 0.0

    def test_an_unpriced_model_returns_none_rather_than_a_guess(self) -> None:
        """Rule 1: no number appears unless something measured it."""
        usage = TokenUsage(input_tokens=1000, output_tokens=100)
        assert cost_usd("some-future-model", usage) is None
