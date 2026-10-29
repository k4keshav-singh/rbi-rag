"""The polite client: rate limiting, retries, and the robots.txt policy.

Every test injects a fake opener and a fake clock, so nothing sleeps and
nothing leaves the machine.
"""

import urllib.error
import urllib.request

import pytest

from rbi_rag.ingest.http import FetchError, PoliteClient


class FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_client(responses: dict[str, object], **kwargs: object) -> tuple[PoliteClient, FakeClock]:
    """Build a client whose opener replays ``responses`` keyed by URL."""
    clock = FakeClock()
    calls: list[str] = []

    def opener(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        calls.append(request.full_url)
        outcome = responses.get(request.full_url, (404, b""))
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if len(outcome) > 1 else outcome[0]
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome  # type: ignore[misc]
        return status, body

    client = PoliteClient(
        user_agent="rbi-rag/test",
        delay_seconds=kwargs.pop("delay_seconds", 2.0),  # type: ignore[arg-type]
        opener=opener,
        sleeper=clock.sleep,
        monotonic=clock,
        **kwargs,  # type: ignore[arg-type]
    )
    client.calls = calls  # type: ignore[attr-defined]
    return client, clock


ROBOTS = "https://example.test/robots.txt"
PAGE = "https://example.test/page"


def test_rejects_a_delay_below_the_floor() -> None:
    with pytest.raises(ValueError, match=r"at least 2\.0"):
        PoliteClient(user_agent="x", delay_seconds=0.5)


def test_waits_between_consecutive_requests() -> None:
    client, clock = make_client({ROBOTS: (404, b""), PAGE: (200, b"ok")})
    client.get(PAGE)
    client.get(PAGE)
    # robots.txt is cached, so the second get makes exactly one more request,
    # and the gap before it must be the full delay.
    assert clock.slept[-1] == pytest.approx(2.0)


def test_a_slow_response_does_not_shorten_the_next_gap() -> None:
    client, clock = make_client({ROBOTS: (404, b""), PAGE: (200, b"ok")})
    client.get(PAGE)
    clock.now += 0.5  # the response took half a second
    client.get(PAGE)
    assert clock.slept[-1] == pytest.approx(1.5)


def test_retries_a_retryable_status_then_succeeds() -> None:
    client, _ = make_client({ROBOTS: (404, b""), PAGE: [(503, b""), (200, b"ok")]})
    assert client.get(PAGE) == b"ok"


def test_gives_up_after_max_retries() -> None:
    client, _ = make_client(
        {ROBOTS: (404, b""), PAGE: (503, b"")},
        max_retries=2,
    )
    with pytest.raises(FetchError, match="after 3 attempts"):
        client.get(PAGE)


def test_does_not_retry_a_client_error() -> None:
    client, _ = make_client({ROBOTS: (404, b""), PAGE: (403, b"")})
    with pytest.raises(FetchError, match="HTTP 403"):
        client.get(PAGE)
    assert client.calls.count(PAGE) == 1  # type: ignore[attr-defined]


def test_backoff_grows() -> None:
    client, clock = make_client(
        {ROBOTS: (404, b""), PAGE: (503, b"")},
        max_retries=3,
    )
    with pytest.raises(FetchError):
        client.get(PAGE)
    backoffs = [s for s in clock.slept if s > 2.0]
    assert backoffs == sorted(backoffs)


class TestRobotsPolicy:
    """RFC 9309: 4xx means no policy published; 5xx means assume disallow."""

    def test_four_hundred_range_allows_access(self) -> None:
        # RBI answers 418 to scripted clients, which lands here.
        client, _ = make_client({ROBOTS: (418, b""), PAGE: (200, b"ok")})
        decision = client.robots_decision(PAGE)
        assert decision.allowed is True
        assert "418" in decision.reason

    def test_five_hundred_range_denies_access(self) -> None:
        client, _ = make_client({ROBOTS: (500, b"")})
        decision = client.robots_decision(PAGE)
        assert decision.allowed is False
        with pytest.raises(FetchError, match=r"robots\.txt disallows"):
            client.get(PAGE)

    def test_network_failure_denies_access(self) -> None:
        client, _ = make_client({ROBOTS: urllib.error.URLError("boom")})
        assert client.robots_decision(PAGE).allowed is False

    def test_an_explicit_disallow_is_honoured(self) -> None:
        body = b"User-agent: *\nDisallow: /page\n"
        client, _ = make_client({ROBOTS: (200, body), PAGE: (200, b"ok")})
        assert client.robots_decision(PAGE).allowed is False

    def test_an_explicit_allow_is_honoured(self) -> None:
        body = b"User-agent: *\nDisallow: /private\n"
        client, _ = make_client({ROBOTS: (200, body), PAGE: (200, b"ok")})
        assert client.robots_decision(PAGE).allowed is True
        assert client.get(PAGE) == b"ok"

    def test_the_decision_is_cached_per_origin(self) -> None:
        client, _ = make_client({ROBOTS: (404, b""), PAGE: (200, b"ok")})
        client.robots_decision(PAGE)
        client.robots_decision(PAGE)
        assert client.calls.count(ROBOTS) == 1  # type: ignore[attr-defined]

    def test_checking_can_be_skipped_deliberately(self) -> None:
        client, _ = make_client({ROBOTS: (500, b""), PAGE: (200, b"ok")})
        assert client.get(PAGE, check_robots=False) == b"ok"
