"""The Final phase's training entrypoint: rebuild a blend from scratch. Every network is retrained from its committed
recipe in configs/members/ (the config it was trained with), the forests from the spec's own `k` and `stride`, then
the blend is saved and the DIII-D submission built in float32. Each step runs as its own `fusioneq.cli` process, so GPU
memory is released between members."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fusioneq.blend import BlendSpec
from fusioneq.settings import CACHE_ROOT, RESULTS

MEMBERS_DIR = Path(__file__).parents[1] / "configs" / "members"
TRAIN_CONFIG = "diii_d_train"
TRAIN_COMMAND = {"mlp": "train", "grid": "train-grid"}


@dataclass(frozen=True)
class Job:
    out: Path | None  # what the step produces; an existing one is skipped (None: always run)
    args: tuple[str, ...]  # fusioneq.cli arguments

    def run(self) -> float:
        t0 = time.time()
        subprocess.run([sys.executable, "-u", "-m", "fusioneq.cli", *self.args], check=True)
        return time.time() - t0


def prepare(config: str) -> Job:
    return Job(CACHE_ROOT / config, ("prepare", "--config", config))


class Rebuild:
    def __init__(self, spec_path: Path, members: Path = MEMBERS_DIR) -> None:
        self.spec_path, self.spec, self.members = spec_path, BlendSpec.from_yaml(spec_path), members

    def train(self, tag: str) -> Job:
        recipe = self.members / f"{tag}.json"
        if not recipe.exists():
            raise FileNotFoundError(f"no committed recipe for {tag}: add {recipe}")
        command = TRAIN_COMMAND[json.loads(recipe.read_text())["kind"]]
        return Job(RESULTS / tag, (command, "--tag", tag, "--config", str(recipe)))

    def jobs(self, configs: list[str], out_dir: Path) -> list[Job]:
        networks, forests, grids = self.spec.members()
        blend = RESULTS / f"blend_{self.spec_path.stem}"
        fit = [
            job
            for t in forests
            for job in (
                Job(
                    RESULTS / t.model,
                    ("fit-trees", "--host", t.host, "--out", t.model, "--k", str(t.k), "--stride", str(t.stride)),
                ),
                Job(RESULTS / Path(t.model).with_suffix(".ubj"), ("compact-trees", t.model)),
            )
        ]
        build = (
            "build",
            "--model",
            str(blend),
            "--configs",
            ",".join(configs),
            "--out-dir",
            str(out_dir),
            "--psi-dtype",
            "float32",
        )
        return [
            prepare(TRAIN_CONFIG),
            *(self.train(t) for t in [*networks, *grids]),  # networks first: each forest reads its host MLP
            *fit,
            Job(blend, ("blend", "--spec", str(self.spec_path), "--tag", blend.name, "--no-score")),
            *(prepare(c) for c in configs),
            Job(None, build),  # never skipped, so a stale submission cannot pass for the rebuilt one
        ]

    def run(self, configs: list[str], out_dir: Path, dry_run: bool = False) -> None:
        log = RESULTS / f"rebuild_{self.spec_path.stem}.jsonl"
        for job in self.jobs(configs, out_dir):
            if job.out and job.out.exists():
                logger.info(f"skip (exists): {job.out}")
                continue
            logger.info(f"{'plan' if dry_run else 'run'}: fusioneq.cli {' '.join(job.args)}")
            if not dry_run:
                seconds = job.run()
                with log.open("a") as f:
                    f.write(json.dumps({"args": job.args, "seconds": round(seconds)}) + "\n")
