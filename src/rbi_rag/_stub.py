"""Entry points for phases that have not landed yet.

These exit non-zero with an explicit message rather than silently succeeding,
so a half-built pipeline can never be mistaken for a working one.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def not_implemented(task: str, phase: int) -> NoReturn:
    """Print where the command is coming from and exit with status 2."""
    sys.stderr.write(f"`{task}` is not implemented yet. It lands in Phase {phase}.\n")
    raise SystemExit(2)
