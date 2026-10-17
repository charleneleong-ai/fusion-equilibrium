"""Official-score progress and metric plots for the PR body, from results/codabench.json.

uv run python scripts/progress_plots.py   # writes experiments/progress/official/*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from autoresearch.timeline import Term, Timeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "progress" / "official"

PLOTS = {
    "ch1_diii_d": Timeline("Ch1: official DIII-D composite S", "d3d_S", (Term("d3d_S", "S"),)),
    "ch1_terms": Timeline(
        "Ch1 terms per DIII-D submission (weights 0.20 / 0.10 / 0.15 / 0.55)",
        "d3d_S",
        (
            Term("d3d_consistency", "consistency"),
            Term("d3d_dlcfs", "boundary_1_minus_d_lcfs", flip=True),
            Term("d3d_r2_qb", "q95_betan_r2"),
            Term("d3d_r2_psi", "flux_r2"),
        ),
    ),
    "ch2_mast_s": Timeline("Ch2: official MAST S (G = S_MAST / S_DIII-D)", "mast_S", (Term("mast_S", "S"),)),
}


def records(path: Path) -> list[dict[str, Any]]:
    """Scored submissions oldest first, each flattened to its scores plus a short commit (or date) name."""
    entries = sorted((e for e in json.loads(path.read_text()) if e["scores"]), key=lambda e: e["submitted"])
    return [{**e["scores"], "name": (e.get("commit") or e["submitted"])[:8]} for e in entries]


def main(path: Path = ROOT / "results" / "codabench.json") -> None:
    recs = records(path)
    for name, timeline in PLOTS.items():
        typer.echo(timeline.plot(recs, OUT / f"{name}.png"))


if __name__ == "__main__":
    typer.run(main)
