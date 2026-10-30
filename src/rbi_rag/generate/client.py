"""The LLM client.

Everything above this module talks to ``LLMClient``, not to the Anthropic SDK,
so the whole generation pipeline and the API can be tested with no network and
no key. ``AnthropicClient`` is the only place that knows the SDK exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rbi_rag.config import Settings
from rbi_rag.schemas import AnswerPayload, TokenUsage


class GenerationError(RuntimeError):
    """The model could not be called, or returned nothing usable."""


@dataclass(frozen=True)
class LLMResult:
    """One completed generation call."""

    payload: AnswerPayload
    usage: TokenUsage
    model: str
    stop_reason: str | None = None


class LLMClient(Protocol):
    """Produces a structured answer from a system prompt and a message turn."""

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> LLMResult: ...


class AnthropicClient:
    """Calls the Messages API and parses the response into AnswerPayload.

    Notes on the current API, each of which is a 400 if got wrong:

    * Structured output goes through ``messages.parse(output_format=Model)``.
      Assistant prefill and forced ``tool_choice`` -- the two older ways to
      coerce JSON -- are both rejected on current models.
    * Thinking cannot be disabled on Claude Sonnet 5.5; depth is controlled
      with ``output_config.effort`` instead. Thinking tokens count against
      ``max_tokens``.
    * ``temperature``/``top_p``/``top_k`` were removed API-wide.
    """

    def __init__(self, settings: Settings) -> None:
        import anthropic

        if settings.anthropic_api_key is None:
            raise GenerationError("ANTHROPIC_API_KEY is not set. Put it in .env; see .env.example.")
        self._settings = settings
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.anthropic_timeout_seconds,
            max_retries=settings.anthropic_max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.anthropic_model

    def complete(self, system: str, messages: Sequence[dict[str, str]]) -> LLMResult:
        try:
            response = self._client.messages.parse(
                model=self._settings.anthropic_model,
                max_tokens=self._settings.anthropic_max_tokens,
                system=system,
                messages=list(messages),  # type: ignore[arg-type]
                output_format=AnswerPayload,
                output_config={"effort": self._settings.anthropic_effort},
            )
        except self._anthropic.AuthenticationError as exc:
            raise GenerationError("Anthropic rejected the API key.") from exc
        except self._anthropic.NotFoundError as exc:
            raise GenerationError(
                f"Model {self._settings.anthropic_model!r} was not found."
            ) from exc
        except self._anthropic.RateLimitError as exc:
            raise GenerationError("Rate limited by Anthropic; retry later.") from exc
        except self._anthropic.APIConnectionError as exc:
            raise GenerationError(f"Could not reach Anthropic: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise GenerationError(f"Anthropic returned HTTP {exc.status_code}.") from exc

        # A safety decline returns 200 with no parsed output, so stop_reason has
        # to be checked before reading content.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise GenerationError(f"The model declined to answer (category: {category}).")

        payload = _first_parsed_output(response)
        if payload is None:
            raise GenerationError(
                f"No structured output in the response (stop_reason={response.stop_reason}). "
                f"A max_tokens stop usually means ANTHROPIC_MAX_TOKENS is too low for "
                f"this effort level, since thinking tokens count against it."
            )

        return LLMResult(
            payload=payload,
            usage=_usage_from(response.usage),
            model=response.model,
            stop_reason=response.stop_reason,
        )


def _first_parsed_output(response: object) -> AnswerPayload | None:
    for block in getattr(response, "content", []):
        parsed = getattr(block, "parsed_output", None)
        if isinstance(parsed, AnswerPayload):
            return parsed
    return None


def _usage_from(usage: object) -> TokenUsage:
    """Read the token counts, tolerating fields a given model may omit."""

    def read(name: str) -> int:
        value = getattr(usage, name, 0)
        return int(value) if isinstance(value, int) else 0

    return TokenUsage(
        input_tokens=read("input_tokens"),
        output_tokens=read("output_tokens"),
        cache_read_tokens=read("cache_read_input_tokens"),
        cache_write_tokens=read("cache_creation_input_tokens"),
    )
