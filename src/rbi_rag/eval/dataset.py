"""Reading and writing the evaluation set.

One rule governs this module: **nothing here may mark an item verified.**
Drafting writes ``verified_by_human=false`` and only the review CLI, acting on
the owner's answer, ever writes true. Every reported metric filters on that
flag, so an automated path that could flip it would quietly turn generated
questions into ground truth.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from rbi_rag.schemas import EvalItem, QuestionType


class UnverifiedItemError(RuntimeError):
    """A script tried to write an item as verified."""


def load_eval_set(path: Path) -> list[EvalItem]:
    """Read the eval set. A missing file is an empty set."""
    if not path.exists():
        return []
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(EvalItem.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} is not a valid eval item") from exc
    return items


def write_eval_set(path: Path, items: Iterable[EvalItem]) -> None:
    """Write the eval set atomically, ordered by id for a stable diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(items, key=lambda item: item.id)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            handle.write(item.model_dump_json())
            handle.write("\n")
    os.replace(temporary, path)


def write_drafted(path: Path, drafted: Iterable[EvalItem]) -> int:
    """Merge drafted candidates in, refusing any that claim verification.

    Existing items win: a re-draft must never overwrite something the owner has
    already reviewed.
    """
    candidates = list(drafted)
    offenders = [item.id for item in candidates if item.verified_by_human]
    if offenders:
        raise UnverifiedItemError(
            f"drafted items may not be verified; offending ids: {', '.join(offenders)}"
        )

    existing = {item.id: item for item in load_eval_set(path)}
    added = 0
    for item in candidates:
        if item.id not in existing:
            existing[item.id] = item
            added += 1
    write_eval_set(path, existing.values())
    return added


def verified_only(items: Sequence[EvalItem]) -> list[EvalItem]:
    """The subset every reported metric is allowed to use."""
    return [item for item in items if item.verified_by_human]


def summarise(items: Sequence[EvalItem]) -> dict[str, int]:
    """Counts by verification state and question type."""
    verified = verified_only(items)
    counts: dict[str, int] = {
        "total": len(items),
        "verified": len(verified),
        "unverified": len(items) - len(verified),
    }
    for question_type in ("factual", "multi_hop", "unanswerable"):
        counts[f"verified_{question_type}"] = sum(
            1 for item in verified if item.question_type == question_type
        )
    return counts


def describe(items: Sequence[EvalItem]) -> str:
    counts = summarise(items)
    types = ", ".join(
        f"{name.removeprefix('verified_')}={counts[name]}"
        for name in counts
        if name.startswith("verified_")
    )
    return (
        f"{counts['total']} item(s): {counts['verified']} verified, "
        f"{counts['unverified']} awaiting review\n"
        f"  verified by type: {types}"
    )


def next_item_id(existing: Sequence[EvalItem], question_type: QuestionType) -> str:
    """A stable, readable id that will not collide with what is already there."""
    prefix = {"factual": "f", "multi_hop": "m", "unanswerable": "u"}[question_type]
    taken = {item.id for item in existing}
    index = 1
    while f"{prefix}{index:04d}" in taken:
        index += 1
    return f"{prefix}{index:04d}"
