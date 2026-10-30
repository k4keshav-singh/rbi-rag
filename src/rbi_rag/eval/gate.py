"""The CI regression gate.

Compares retrieval on the committed snapshot against the committed baseline
and fails when a metric has dropped by more than a configurable margin.

The comparison is a pure function over two dictionaries, so the part that
decides pass or fail is unit-tested with no database, no model and no network.
Only the part that produces the current numbers needs Postgres.

What the gate is **not**: a statement about retrieval quality. It runs on a
fixture corpus with fixture questions. A drop means "something changed since
the baseline was recorded", which is worth blocking a merge over; the absolute
value means nothing outside this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: Arms the gate measures. Reranking is excluded: the cross-encoder is a model
#: download, and the gate is deliberately hermetic.
GATE_RETRIEVERS = ("dense", "fts", "hybrid")


@dataclass(frozen=True)
class Finding:
    """One baseline-versus-current comparison."""

    arm: str
    metric: str
    baseline: float
    current: float | None
    max_drop: float

    @property
    def delta(self) -> float | None:
        if self.current is None:
            return None
        return self.current - self.baseline

    @property
    def failed(self) -> bool:
        # Unmeasurable is a failure: the gate cannot confirm nothing regressed.
        if self.current is None:
            return True
        return self.current < self.baseline - self.max_drop

    @property
    def improved(self) -> bool:
        delta = self.delta
        return delta is not None and delta > self.max_drop

    def describe(self) -> str:
        if self.current is None:
            return f"  FAIL {self.arm} {self.metric}: not measurable (baseline {self.baseline:.4f})"
        marker = "FAIL" if self.failed else ("UP  " if self.improved else "ok  ")
        delta = self.delta or 0.0
        return (
            f"  {marker} {self.arm} {self.metric}: "
            f"{self.current:.4f} vs baseline {self.baseline:.4f} ({delta:+.4f})"
        )


@dataclass
class GateReport:
    """Everything the gate decided, and why."""

    findings: list[Finding] = field(default_factory=list)
    missing_arms: list[str] = field(default_factory=list)
    unexpected_arms: list[str] = field(default_factory=list)
    max_drop: float = 0.05

    @property
    def failures(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.failed]

    @property
    def improvements(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.improved]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.missing_arms

    def describe(self) -> str:
        lines = [
            f"retrieval gate: {len(self.findings)} comparison(s), tolerance {self.max_drop:.4f}"
        ]
        lines.extend(finding.describe() for finding in self.findings)

        for arm in self.missing_arms:
            lines.append(f"  FAIL {arm}: in the baseline but not measured in this run")
        for arm in self.unexpected_arms:
            lines.append(f"  note {arm}: measured but absent from the baseline; not gated")

        if self.failures:
            lines.append("")
            lines.append(
                f"FAILED: {len(self.failures)} metric(s) dropped by more than "
                f"{self.max_drop:.4f}. If the change is intended, re-record the "
                f"baseline with `uv run poe eval snapshot --baseline-only` and say "
                f"why in the commit message."
            )
        elif self.improvements:
            lines.append("")
            lines.append(
                f"PASSED, and {len(self.improvements)} metric(s) improved. Consider "
                f"re-recording the baseline so future regressions are measured from "
                f"the better number."
            )
        else:
            lines.append("")
            lines.append("PASSED: nothing regressed.")
        return "\n".join(lines)


def compare(
    baseline: Mapping[str, Mapping[str, float]],
    current: Mapping[str, Mapping[str, float | None]],
    *,
    metrics: Sequence[str] = ("recall@1", "recall@5", "mrr"),
    max_drop: float = 0.05,
) -> GateReport:
    """Decide whether this run regressed against the baseline.

    Every arm in the baseline must appear in the current run: an arm that
    quietly stopped being measured is a regression in the gate itself, not a
    pass. Arms present only in the current run are reported but not gated,
    since there is nothing to compare them against.
    """
    if max_drop < 0:
        raise ValueError("max_drop must not be negative")

    report = GateReport(max_drop=max_drop)

    for arm in sorted(baseline):
        if arm not in current:
            report.missing_arms.append(arm)
            continue
        for metric in metrics:
            if metric not in baseline[arm]:
                continue
            report.findings.append(
                Finding(
                    arm=arm,
                    metric=metric,
                    baseline=float(baseline[arm][metric]),
                    current=current[arm].get(metric),
                    max_drop=max_drop,
                )
            )

    report.unexpected_arms = sorted(set(current) - set(baseline))
    return report


def baseline_from(results: Mapping[str, Mapping[str, float | None]]) -> dict[str, dict[str, float]]:
    """Turn a measured run into a committable baseline.

    Unmeasurable metrics are dropped rather than recorded as zero: a baseline
    of 0.0 would make any future result look like an improvement and the gate
    would never fire again.
    """
    recorded: dict[str, dict[str, float]] = {}
    for arm, metrics in results.items():
        kept = {name: float(value) for name, value in metrics.items() if value is not None}
        if kept:
            recorded[arm] = kept
    return recorded


def arms_for(
    strategy: str, target_tokens: int, retrievers: Sequence[str] = GATE_RETRIEVERS
) -> list[str]:
    """Arm ids the gate measures for a snapshot's chunk configuration."""
    return [f"{strategy}-{target_tokens}-{retriever}-norerank" for retriever in retrievers]
