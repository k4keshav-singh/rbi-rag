"""Content-addressed cache for LLM responses.

The ablation is supposed to be reproducible, but an LLM is not deterministic.
Caching by the hash of everything that went into a call gets most of the way
there: a re-run of an unchanged configuration replays the same responses, costs
nothing, and produces the same table. Change the prompt, the model or the
retrieved context and the key changes, so a stale answer can never be served
for a different question.

What this does **not** do is make generation metrics deterministic in the sense
retrieval metrics are. A cleared cache re-samples the model. The results header
says so rather than implying otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(**parts: Any) -> str:
    """Hash of every input that could change the response.

    Keys are sorted so that an unordered dict cannot produce two different
    hashes for the same call.
    """
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """A directory of JSON files, one per call."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by the first two characters: a flat directory of tens of
        # thousands of files is slow to list on Windows.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt entry is a cache miss, never a crash.
            self.misses += 1
            return None
        self.hits += 1
        return loaded

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

    def describe(self) -> str:
        total = self.hits + self.misses
        if not self.enabled:
            return "cache disabled"
        if total == 0:
            return "cache unused"
        return f"cache {self.hits}/{total} hits ({self.hits / total:.0%})"
