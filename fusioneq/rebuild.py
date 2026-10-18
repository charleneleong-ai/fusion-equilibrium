"""The Final phase's training entrypoint: rebuild a blend from scratch. Every network is retrained from its committed
recipe in configs/members/ (the config it was trained with), the forests from the spec's own `k` and `stride`, then
the blend is saved and the DIII-D submission built in float32. For a MAST config it also fits the DIII-D
calibrations the MAST path reads and builds the MAST flux and scalars from the MAST inputs alone. Each step runs as its
own `fusioneq.cli` process, so GPU memory is released between members."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fusioneq.blend import BlendSpec
from fusioneq.finish import FinishSpec
from fusioneq.settings import CACHE_ROOT, RESULTS

MEMBERS_DIR = Path(__file__).parents[1] / "configs" / "members"
TRAIN_CONFIG = "diii_d_train"
TRAIN_COMMAND = {"mlp": "train", "grid": "train-grid"}
# The MAST path's settings; none was chosen on MAST labels (organisers' rule, 23 Sep forum post). kappa, alpha and frac
# are the code defaults from before the 1 Oct demo-shot sweep, whose picks (alpha 2.0, frac 0.4) are dropped; the
# aspect-ratio cap 0.65 was chosen on our own public-fold scores.
MAST_KAPPA, MAST_ALPHA, MAST_FRAC, MAST_EPS = "1.6", "1.0", "0.25", "0.65"
MAST_FLUX, MAST_SCAL, MAST_PSI = (
    f"psi_k{MAST_KAPPA}_a{MAST_ALPHA}_f{MAST_FRAC}.npy",
    "scal_clean.npy",
    "psi_shape_clean.npy",
)


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


def calibrations() -> list[Job]:
    """What the MAST path reads that is fitted on DIII-D labels: the coil and plasma calibration, the betaN pressure
    calibration, the scorer-derived shape targets, then the shape map and the Shafranov-shift model."""
    return [
        Job(CACHE_ROOT / "phys_calib.json", ("physics-check",)),
        Job(CACHE_ROOT / "scalar_calib_volume.json", ("scalar-check", "--pressure", "volume")),
        Job(CACHE_ROOT / TRAIN_CONFIG / "cons_targets.npz", ("shape-targets",)),
        Job(CACHE_ROOT / "shape_map.json", ("shape-fit",)),
        Job(CACHE_ROOT / "shift_model.json", ("shift-fit",)),
    ]


def mast(config: str) -> list[Job]:
    """MAST flux and q95/betaN from the MAST inputs: physics flux (which also writes the Thomson geometry), scalars on
    its boundary, then the shape-map flux with the Shafranov-shifted axis that is submitted."""
    d, c = CACHE_ROOT / config, ("--config", config)
    flux = ("--kappa", MAST_KAPPA, "--alpha", MAST_ALPHA, "--frac", MAST_FRAC, "--out", MAST_FLUX)
    scal = ("--psi-file", MAST_FLUX, "--eps-max", MAST_EPS, "--pressure", "volume", "--out", MAST_SCAL)
    return [
        Job(d / MAST_FLUX, ("mast-physics", *c, *flux)),
        Job(d / MAST_SCAL, ("mast-scalars", *c, *scal)),
        Job(d / MAST_PSI, ("mast-shape", *c, "--out", MAST_PSI, "--shift-axis", "--alpha", MAST_ALPHA)),
    ]


def finishing(recipe: Path) -> list[Job]:
    """The DIII-D finishing step's training: the label scalars on every `stride`-th training frame, then each seed of
    the heads (docs/finishing.md)."""
    s = FinishSpec.load(recipe)
    targets = CACHE_ROOT / TRAIN_CONFIG / f"finish_targets_s{s.stride}.npz"
    heads = [
        Job(RESULTS / tag, ("train-heads", "--spec", str(recipe), "--seed", str(seed)))
        for seed, tag in zip(s.seeds, s.head_tags(), strict=True)
    ]
    return [Job(targets, ("finish-targets", "--stride", str(s.stride))), *heads]


class Rebuild:
    def __init__(self, spec_path: Path, members: Path = MEMBERS_DIR, finish: Path | None = None) -> None:
        self.spec_path, self.spec, self.members = spec_path, BlendSpec.from_yaml(spec_path), members
        self.finish = finish

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
        masts = [c for c in configs if c.startswith("mast")]
        if masts:
            build += ("--mast-psi", MAST_PSI, "--mast-scal", MAST_SCAL)
        if self.finish is not None:
            build += ("--finish", str(self.finish))
        return [
            prepare(TRAIN_CONFIG),
            *(self.train(t) for t in [*networks, *grids]),  # networks first: each forest reads its host MLP
            *fit,
            Job(blend, ("blend", "--spec", str(self.spec_path), "--tag", blend.name, "--no-score")),
            *(finishing(self.finish) if self.finish is not None else []),
            *(calibrations() if masts else []),
            *(prepare(c) for c in configs),
            *(job for c in masts for job in mast(c)),
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
