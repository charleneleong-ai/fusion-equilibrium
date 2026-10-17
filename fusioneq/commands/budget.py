"""Final-phase rebuild budget of blend specs (fusioneq.budget)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from fusioneq.blend import BlendSpec
from fusioneq.budget import LIMIT_HOURS, RebuildBudget


def rebuild_budget(
    specs: Annotated[list[Path], typer.Argument(help="blend specs, e.g. configs/blends/*.yaml")],
) -> None:
    """Idle-A100 training hours (low-high) of each blend against the Final's 24 GPU-hour rebuild cap."""
    for path in specs:
        b = RebuildBudget.of(BlendSpec.from_yaml(path))
        lo, hi = b.hours()
        verdict = "fits" if hi <= LIMIT_HOURS else "OVER" if lo > LIMIT_HOURS else "at risk"
        logger.info(
            f"{path.stem}: {b.mlps} MLPs, {b.forests} forests, grids {list(b.grids)}: "
            f"{lo:.1f}-{hi:.1f} A100-h of {LIMIT_HOURS:.0f} ({verdict})"
        )


COMMANDS = [rebuild_budget]
