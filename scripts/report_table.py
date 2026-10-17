"""Emit the LaTeX results rows for report/main.tex from results/*_score.json (fetched from the A100)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
MARK_START, MARK_END = "    % results-table-start", "    % results-table-end"
PLACEHOLDER = "    % filled from results/*_score.json by scripts/report_table.py"


def row(d: dict) -> str:
    tag = d["tag"].replace("_", "\\_")
    one_minus = 1 - min(1.0, d["dlcfs"])
    cells = [
        f"{d['S']:.4f}",
        f"{max(0, d['r2_psi']):.4f}",
        f"{max(0, d['r2_qb']):.4f}",
        f"{one_minus:.4f}",
        f"{d['consistency']:.4f}",
    ]
    return "    " + " & ".join([tag, *cells]) + " \\\\"


def main(results: Path = ROOT / "results", tex: Path = ROOT / "report" / "main.tex", write: bool = True) -> None:
    scores = sorted((json.loads(p.read_text()) for p in results.glob("*_score.json")), key=lambda d: -d["S"])
    rows = "\n".join(row(d) for d in scores)
    print(rows)
    if not write:
        return
    s = tex.read_text().replace(PLACEHOLDER, f"{MARK_START}\n{MARK_END}")
    head, rest = s.split(MARK_START)
    _, tail = rest.split(MARK_END)
    tex.write_text(f"{head}{MARK_START}\n{rows}\n{MARK_END}{tail}")


if __name__ == "__main__":
    typer.run(main)
