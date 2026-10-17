"""Render experiments/progress/mlp/progress.png: local DIII-D S of every sweep run (best so far as a line) and the
official Codabench scores in submission order (DIII-D S, MAST S, G_ratio).

    uv run python scripts/progress_chart.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = re.compile(r"^(v\d+)_")
ORDINAL = re.compile(r"\((\d+)(?:st|nd|rd|th)")


def sweep_runs(results: Path) -> list[tuple[str, float]]:
    """(sweep, S) of every sweep run (PCA-MLP runs and ensembles: the v<N>_ tags), in sweep order."""
    runs = []
    for path in results.glob("v*_score.json"):
        d: dict[str, Any] = json.loads(path.read_text())
        m = SWEEP.match(d["tag"])
        if m and d.get("S") is not None:
            runs.append((m.group(1), float(d["S"])))
    return sorted(runs, key=lambda r: int(r[0][1:]))


def submissions(codabench: Path) -> list[dict[str, float]]:
    """Scored official submissions in upload order."""
    scored = [e for e in json.loads(codabench.read_text()) if e.get("scores")]
    order = [int(m.group(1)) if (m := ORDINAL.search(e["submitted"])) else 1 for e in scored]
    return [e["scores"] for _, e in sorted(zip(order, scored, strict=True), key=lambda p: p[0])]


def main(
    results: Path = ROOT / "results", out: Path = ROOT / "experiments" / "progress" / "mlp" / "progress.png"
) -> None:
    runs, subs = sweep_runs(results), submissions(results / "codabench.json")
    sweeps = sorted({s for s, _ in runs}, key=lambda s: int(s[1:]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = [sweeps.index(s) for s, _ in runs]
    ax1.scatter(x, [v for _, v in runs], s=18, alpha=0.6, label="run / ensemble")
    best = [max(v for s, v in runs if sweeps.index(s) <= i) for i in range(len(sweeps))]
    ax1.plot(range(len(sweeps)), best, "k-", lw=1.5, label="best so far")
    ax1.set(xticks=range(len(sweeps)), xticklabels=sweeps, ylim=(0.96, 0.995))
    ax1.set_ylabel("local DIII-D S (40 held-out shots)")
    ax1.set_title("DIII-D: local score per sweep")
    ax1.legend(loc="lower right")
    n = range(1, len(subs) + 1)
    ax2.plot(n, [s["d3d_S"] for s in subs], "o-", color="C0", label="DIII-D S")
    ax2.set(xlabel="official submission", ylabel="DIII-D S", ylim=(0.989, 0.993))
    twin = ax2.twinx()
    twin.plot(n, [s["mast_S"] for s in subs], "s--", color="C3", label="MAST S")
    twin.plot(n, [s["g_ratio"] for s in subs], "^:", color="C2", label="G_ratio")
    twin.set(ylabel="MAST S, G_ratio", ylim=(0.0, 0.7))
    ax2.set_title("Codabench: official scores")
    lines = ax2.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax2.legend(lines[0] + extra[0], lines[1] + extra[1], loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    typer.echo(f"wrote {out}: {len(runs)} runs over {len(sweeps)} sweeps, {len(subs)} official submissions")


if __name__ == "__main__":
    typer.run(main)
