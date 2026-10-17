"""Schedule-driven sweep orchestrator for fusioneq, built on the `autoresearch` package.

Reads `configs/schedules/<name>.yaml` (`common_overrides` + `iters`), launches one
`python -m fusioneq.cli train` subprocess per iter under GPU triage, and appends a
BASELINE / KEEP / DISCARD / EARLY_KILL / CRASH row to
`experiments/fusioneq/<config>/results.jsonl`. Score is the official composite S on the
held-out shots. The per-iter W&B run URL travels in the row.

    python -m experiments.autoresearch run --schedule configs/schedules/v1_mlp.yaml --config mlp
    python -m experiments.autoresearch show --config mlp
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from autoresearch.gpu_monitor import GPUTriageThresholds
from autoresearch.results import decide_status, load_results
from autoresearch.sweep_runner import GPUTriageMonitor, IterPlan, SweepRunner
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
TAG = "fusioneq"
RESULTS_DIR = Path("~/fusion/results").expanduser()
app = typer.Typer(add_completion=False)


@dataclass
class Schedule:
    name: str
    command: str
    common_overrides: list[str]
    iters: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> Schedule:
        d = yaml.safe_load(path.read_text())
        return cls(path.stem, d.get("command", "train"), [str(x) for x in d.get("common_overrides", [])], d["iters"])


class SchedulePlanner:
    """One IterPlan per schedule entry, skipping iters whose tag already has a row (resumable)."""

    def __init__(self, schedule: Schedule, config: str, python: str, results_dir: Path) -> None:
        self.schedule, self.config, self.python, self.results_dir = schedule, config, python, results_dir

    def iter_tag(self, i: int) -> str:
        return f"{self.schedule.name}_i{i:02d}"

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        done = {r.get("description", "").split(" ")[0] for r in history}
        for i, it in enumerate(self.schedule.iters):
            tag = self.iter_tag(i)
            if tag in done:
                continue
            overrides = [str(x) for x in it.get("overrides", [])]
            # an iter may run another command (e.g. ensemble); the common overrides belong to the schedule's command
            common = self.schedule.common_overrides if "command" not in it else []
            cmd = [
                self.python,
                "-m",
                "fusioneq.cli",
                it.get("command", self.schedule.command),
                *common,
                *overrides,
                "--tag",
                tag,
            ]
            yield IterPlan(
                cmd=cmd,
                description=f"{tag} {it.get('description', '')}".strip(),
                config_name=self.config,
                notes=" ".join(overrides),
                env={**os.environ, "FUSIONEQ_RESULTS": str(self.results_dir), "PYTHONPATH": str(ROOT)},
                cwd=ROOT,
                timeout_min=int(it.get("timeout_min", 180)),
            )


class ScoreJsonExtractor:
    """Turn the train command's `<tag>_score.json` into a results row; missing file means CRASH."""

    def __init__(self, results_dir: Path, config: str, experiments_dir: Path = ROOT / "experiments") -> None:
        self.results_dir, self.config, self.experiments_dir = results_dir, config, experiments_dir

    def extract(self, plan: IterPlan, run_id: str | None, exit_code: int) -> list[dict[str, Any]]:
        tag = plan.description.split(" ")[0]
        path = self.results_dir / f"{tag}_score.json"
        if exit_code != 0 or not path.exists():
            return [
                {
                    "score": 0.0,
                    "status": "CRASH",
                    "description": plan.description,
                    "config_name": self.config,
                    "notes": plan.notes,
                }
            ]
        d = json.loads(path.read_text())
        score = float(d["S"])
        status = decide_status(load_results(self.experiments_dir, TAG, self.config), score)
        metrics = {k: d[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "lcfs_fail_frac") if k in d}
        return [
            {
                "score": score,
                "status": status,
                "description": plan.description,
                "config_name": self.config,
                "notes": plan.notes,
                "wandb_url": d.get("wandb_url", ""),
                "runtime_min": float(d.get("runtime_min", 0.0)),
                "steps": int(d.get("epochs", 0)),
                "metrics": metrics,
                "r2_cons_each": d.get("r2_cons_each", {}),
            }
        ]


@app.command()
def run(
    schedule: Annotated[Path, typer.Option(help="configs/schedules/<name>.yaml")],
    config: str = "mlp",
    python: str = sys.executable,
    results_dir: Path = RESULTS_DIR,
    iter_timeout_min: int = 180,
) -> None:
    sched = Schedule.load(schedule)
    logger.info(f"{sched.name}: {len(sched.iters)} iters, common {sched.common_overrides}")
    runner = SweepRunner(
        tag=TAG,
        planner=SchedulePlanner(sched, config, python, results_dir),
        extractor=ScoreJsonExtractor(results_dir, config),
        # This model is small by design (<12 GB of 80): the undersized-memory rule killed two healthy v4 iters.
        triage=GPUTriageMonitor(GPUTriageThresholds(undersized_mem_pct=0)),
        experiments_dir=ROOT / "experiments",
        iter_timeout_min=iter_timeout_min,
        pause_between_iters_s=5,
    )
    result = runner.run()
    logger.info(f"{sched.name}: {result.iterations} iters, {result.kills} kills, blocked={result.blocked}")


@app.command()
def show(config: str = "mlp") -> None:
    rows = load_results(ROOT / "experiments", TAG, config)
    for r in rows:
        m = r.get("metrics", {})
        logger.info(
            f"E{r['experiment']:02d} {r['status']:9s} S={r['score']:.4f} r2psi={m.get('r2_psi', 0):.4f} "
            f"cons={m.get('consistency', 0):.4f} {r['description']} {r.get('wandb_url', '')}"
        )


if __name__ == "__main__":
    app()
