"""Recompute BASELINE/KEEP/DISCARD for every row of a results.jsonl against all rows before it (CRASH kept)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from autoresearch.results import decide_status


def main(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if row.get("status") in ("CRASH", "EARLY_KILL"):
            continue
        row["status"] = decide_status(rows[:i], float(row["score"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(f"{r['status']:9s} {r['score']:.4f} {r['description'][:60]}")


if __name__ == "__main__":
    typer.run(main)
