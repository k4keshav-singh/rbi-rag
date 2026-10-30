"""Calibrating the faithfulness judge against the owner's labels.

A faithfulness score from an unmeasured judge is a statement about the judge.
This module produces the number that has to appear beside it: how often the
judge and a human agree, corrected for the agreement you would get by chance.

Chance correction matters here specifically. Most answers are faithful, so a
judge that said "faithful" unconditionally would score around 90% raw
agreement and be worthless. Kappa near zero exposes exactly that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rbi_rag.eval.metrics import cohens_kappa

#: Landis & Koch (1977) bands. A naming convention for reading kappa, not a
#: measurement and not a pass mark.
_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
)


def interpret(kappa: float) -> str:
    for threshold, label in _BANDS:
        if kappa >= threshold:
            return label
    return "poor"


@dataclass(frozen=True)
class CalibrationReport:
    """Agreement between the judge and the owner on a labelled subset."""

    n: int
    observed_agreement: float | None
    kappa: float | None
    judge_positive_rate: float | None
    human_positive_rate: float | None
    degenerate: bool

    def describe(self) -> str:
        if self.n == 0:
            return (
                "judge calibration: TODO - no human labels yet. Faithfulness "
                "is reported without calibration and should be read with that "
                "in mind."
            )
        if self.kappa is None:
            return f"judge calibration: {self.n} label(s), too few to compute kappa (need 2)."

        lines = [
            f"judge calibration on {self.n} item(s): "
            f"kappa={self.kappa:.3f} ({interpret(self.kappa)}), "
            f"raw agreement={self.observed_agreement:.0%}"
        ]
        if self.judge_positive_rate is not None and self.human_positive_rate is not None:
            lines.append(
                f"  judge called {self.judge_positive_rate:.0%} faithful; "
                f"owner called {self.human_positive_rate:.0%} faithful"
            )
        if self.degenerate:
            lines.append(
                "  WARNING: one or both labellers were constant across the subset, "
                "so kappa is degenerate here. Label some answers you believe are "
                "unfaithful before trusting it."
            )
        return "\n".join(lines)


def calibrate(
    judge_labels: Sequence[bool],
    human_labels: Sequence[bool],
) -> CalibrationReport:
    """Compare judge verdicts with the owner's on the same items."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge and human label counts differ")

    total = len(judge_labels)
    if total == 0:
        return CalibrationReport(0, None, None, None, None, degenerate=False)

    observed = sum(1 for a, b in zip(judge_labels, human_labels, strict=True) if a == b) / total
    judge_rate = sum(judge_labels) / total
    human_rate = sum(human_labels) / total
    constant = len(set(judge_labels)) == 1 or len(set(human_labels)) == 1

    return CalibrationReport(
        n=total,
        observed_agreement=observed,
        kappa=cohens_kappa(judge_labels, human_labels),
        judge_positive_rate=judge_rate,
        human_positive_rate=human_rate,
        degenerate=constant,
    )
