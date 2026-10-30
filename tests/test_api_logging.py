"""JSON log formatting."""

import json
import logging

from rbi_rag.api.logging import JsonFormatter


def record(message: str = "hello", **extra: object) -> logging.LogRecord:
    made = logging.LogRecord(
        name="rbi_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(made, key, value)
    return made


def formatted(**extra: object) -> dict[str, object]:
    return json.loads(JsonFormatter().format(record(**extra)))


class TestJsonFormatter:
    def test_emits_one_json_object(self) -> None:
        output = JsonFormatter().format(record())
        assert "\n" not in output
        assert json.loads(output)["message"] == "hello"

    def test_includes_level_logger_and_timestamp(self) -> None:
        payload = formatted()
        assert payload["level"] == "INFO"
        assert payload["logger"] == "rbi_rag.test"
        assert payload["ts"]

    def test_extra_fields_are_promoted(self) -> None:
        """Per-request latency and cost are logged via `extra`."""
        payload = formatted(latency_ms=12.5, cost_usd=0.002, abstained=False)
        assert payload["latency_ms"] == 12.5
        assert payload["cost_usd"] == 0.002
        assert payload["abstained"] is False

    def test_non_json_values_are_stringified_rather_than_crashing(self) -> None:
        payload = formatted(obj=object())
        assert isinstance(payload["obj"], str)

    def test_nested_structures_survive(self) -> None:
        payload = formatted(violations=["a", "b"], meta={"k": 1})
        assert payload["violations"] == ["a", "b"]
        assert payload["meta"] == {"k": 1}

    def test_exceptions_are_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            made = record()
            made.exc_info = sys.exc_info()
            payload = json.loads(JsonFormatter().format(made))
        assert "ValueError" in payload["exception"]
