"""Writing results tables.

Every table carries the git SHA, the date, the chunk-set configuration and the
size of the verified eval set it was computed from, because a metric without
its provenance cannot be reproduced or trusted later.

Numbers that were not measured are written as ``TODO``, never as a blank that
reads like zero and never as an estimate.
"""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TODO = "TODO"


def git_sha() -> str | None:
    """The commit a result was produced from."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


@dataclass
class ArmResult:
    """One cell of the ablation: a configuration and what it scored."""

    strategy: str
    target_tokens: int
    retriever: str
    rerank: bool
    chunk_set_id: int
    items: int = 0
    truncated_chunks: int = 0
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def arm_id(self) -> str:
        rerank = "rerank" if self.rerank else "norerank"
        return f"{self.strategy}-{self.target_tokens}-{self.retriever}-{rerank}"


def format_value(value: Any) -> str:
    """Render a metric. Unmeasured is TODO, never a blank or a zero."""
    if value is None:
        return TODO
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _header(title: str, context: dict[str, Any]) -> list[str]:
    lines = [f"# {title}", ""]
    for key, value in context.items():
        lines.append(f"- **{key}**: {format_value(value)}")
    lines.append("")
    return lines


def render_markdown(
    title: str,
    context: dict[str, Any],
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
    notes: Sequence[str] = (),
) -> str:
    """One markdown table with its provenance above and caveats below."""
    lines = _header(title, context)

    if not rows:
        lines.append("_No rows: nothing measurable yet._")
        lines.append("")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_value(row.get(c)) for c in columns) + " |")
        lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """CSV alongside the markdown, for anyone who wants to plot it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})


def arm_rows(results: Sequence[ArmResult], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "arm": result.arm_id,
            "strategy": result.strategy,
            "chunk_size": result.target_tokens,
            "retriever": result.retriever,
            "rerank": result.rerank,
            "chunk_set_id": result.chunk_set_id,
            "items": result.items,
            "truncated_chunks": result.truncated_chunks,
        }
        for name in metric_names:
            row[name] = result.metrics.get(name)
        rows.append(row)
    return rows


def run_context(
    *,
    verified_items: int,
    k_values: Sequence[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "generated": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "verified_eval_items": verified_items,
        "k_values": ", ".join(str(k) for k in k_values),
    }
    if extra:
        context.update(extra)
    return context


def as_dict(result: ArmResult) -> dict[str, Any]:
    return asdict(result)
