"""A deliberately slow, well-behaved HTTP client for rbi.org.in.

Built on the standard library so that the downloader adds no dependency. The
rate limit, the retry policy and the robots.txt check are all injectable, which
is what lets the tests cover them without touching the network or sleeping.
"""

from __future__ import annotations

import random
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

#: Status codes worth retrying: transient server trouble and explicit throttling.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

Opener = Callable[[urllib.request.Request, float], tuple[int, bytes]]
Sleeper = Callable[[float], None]


class FetchError(RuntimeError):
    """A request failed in a way that retrying will not fix."""


@dataclass(frozen=True)
class RobotsDecision:
    """Why we believe a URL may or may not be fetched."""

    allowed: bool
    reason: str


def _default_opener(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(response.status)
        return status, bytes(response.read())


class PoliteClient:
    """Issues GET requests no faster than ``delay_seconds`` apart.

    The delay is enforced between the *start* of consecutive requests, so a slow
    response never shortens the gap to the next one.
    """

    def __init__(
        self,
        user_agent: str,
        delay_seconds: float,
        max_retries: int = 5,
        timeout: float = 60.0,
        opener: Opener | None = None,
        sleeper: Sleeper | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if delay_seconds < 2.0:
            raise ValueError("delay_seconds must be at least 2.0; see Settings.")
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._open = opener or _default_opener
        self._sleep = sleeper or time.sleep
        self._now = monotonic or time.monotonic
        self._last_request_at: float | None = None
        self._robots: dict[str, RobotsDecision | None] = {}

    # -- rate limiting ----------------------------------------------------

    def _wait_turn(self) -> None:
        if self._last_request_at is not None:
            elapsed = self._now() - self._last_request_at
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._now()

    # -- robots.txt -------------------------------------------------------

    def robots_decision(self, url: str) -> RobotsDecision:
        """Decide whether ``url`` may be fetched, per RFC 9309.

        A 4xx on robots.txt means "no restrictions published", so access is
        allowed. A 5xx or a network failure means we must assume the whole site
        is disallowed. RBI currently answers 418 to scripted clients, which is a
        4xx and therefore the permissive branch -- but we record that we never
        actually read a policy rather than claiming we honoured one.
        """
        origin = self._origin(url)
        cached = self._robots.get(origin)
        if cached is not None:
            return cached

        robots_url = urljoin(origin, "/robots.txt")
        try:
            status, body = self._request_once(robots_url)
        except urllib.error.HTTPError as exc:
            decision = (
                RobotsDecision(True, f"robots.txt returned {exc.code}; no policy published")
                if 400 <= exc.code < 500
                else RobotsDecision(False, f"robots.txt returned {exc.code}; assuming disallow")
            )
            self._robots[origin] = decision
            return decision
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            decision = RobotsDecision(False, f"robots.txt unreachable ({exc}); assuming disallow")
            self._robots[origin] = decision
            return decision

        if 400 <= status < 500:
            decision = RobotsDecision(True, f"robots.txt returned {status}; no policy published")
        elif status >= 500:
            decision = RobotsDecision(False, f"robots.txt returned {status}; assuming disallow")
        else:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(body.decode("utf-8", errors="replace").splitlines())
            allowed = parser.can_fetch(self.user_agent, url)
            decision = RobotsDecision(allowed, "robots.txt parsed")
        self._robots[origin] = decision
        return decision

    @staticmethod
    def _origin(url: str) -> str:
        parts = urlparse(url)
        return f"{parts.scheme}://{parts.netloc}"

    # -- requests ---------------------------------------------------------

    def _request_once(self, url: str) -> tuple[int, bytes]:
        self._wait_turn()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        return self._open(request, self.timeout)

    def get(self, url: str, *, check_robots: bool = True) -> bytes:
        """Fetch ``url``, retrying transient failures with exponential backoff."""
        if check_robots:
            decision = self.robots_decision(url)
            if not decision.allowed:
                raise FetchError(f"robots.txt disallows {url}: {decision.reason}")

        last_error: str = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                status, body = self._request_once(url)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS:
                    raise FetchError(f"GET {url} failed with HTTP {exc.code}") from exc
                last_error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
            else:
                if status in RETRYABLE_STATUS:
                    last_error = f"HTTP {status}"
                elif status >= 400:
                    raise FetchError(f"GET {url} failed with HTTP {status}")
                else:
                    return body

            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt))

        raise FetchError(f"GET {url} failed after {self.max_retries + 1} attempts: {last_error}")

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, on top of the standing rate limit."""
        base = min(2.0**attempt, 30.0)
        return base + random.uniform(0.0, 1.0)
