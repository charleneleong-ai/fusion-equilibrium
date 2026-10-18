"""The DIII-D finishing step (fusioneq/finish.py, docs/finishing.md).

finish-targets  the scorer's seven scalars of the label flux on every `stride`-th DIII-D training frame
train-heads     one seed of a finishing recipe's scalar heads, trained on those frames
finish          an existing DIII-D submission .npz finished toward the recipe's heads (build --finish does the same)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from fusioneq.cache import Frames
from fusioneq.commands.submission import Submission
from fusioneq.consistency import scorer_scalars
from fusioneq.evaluation import HeldOut
from fusioneq.finish import FinishSpec, finish_frames
from fusioneq.scalar_heads import HeadConfig, ScalarHeads
from fusioneq.settings import CACHE_ROOT, RESULTS

SpecOption = Annotated[Path, typer.Option(help="finishing recipe, e.g. configs/finish/axis_li_v1.json")]


def targets_path(cache_root: Path, stride: int) -> Path:
    return cache_root / "diii_d_train" / f"finish_targets_s{stride}.npz"


def finish_targets(stride: int = 1, cache_root: Path = CACHE_ROOT, workers: int = 12) -> None:
    """finish_targets_s<stride>.npz: the scorer's scalars of the label flux on every `stride`-th training frame of
    the fixed 200-shot hold-out split (held-out shots never enter), reusing cons_targets.npz where it has them."""
    ev = HeldOut.load(cache_root)
    rows = np.nonzero(ev.train)[0][::stride]
    y = np.full((len(rows), 7), np.nan)
    cons = cache_root / "diii_d_train" / "cons_targets.npz"
    todo = np.ones(len(rows), bool)
    if cons.exists():
        z = np.load(cons)
        at = np.searchsorted(rows, z["train_rows"]).clip(max=len(rows) - 1)
        hit = rows[at] == z["train_rows"]
        y[at[hit]], todo[at[hit]] = z["train"][hit], False
    logger.info(f"{len(rows)} frames, {todo.sum()} to derive")
    with ProcessPoolExecutor(workers) as ex:
        y[todo] = np.stack(list(ex.map(scorer_scalars, (ev.fr.psi[r] for r in rows[todo]), chunksize=512)))
    np.savez(targets_path(cache_root, stride), rows=rows, y=y)


def train_heads(spec: SpecOption, seed: int = 0, cache_root: Path = CACHE_ROOT) -> None:
    """Train one seed of the recipe's heads on finish_targets_s<stride>.npz; saves results/heads_<recipe>_s<seed>."""
    s = FinishSpec.load(spec)
    z = np.load(targets_path(cache_root, s.stride))
    fr = Frames.load(cache_root / "diii_d_train")
    t0 = time.time()
    heads = ScalarHeads(HeadConfig(**{**s.heads, "seed": seed})).fit(fr, z["rows"], z["y"])
    heads.save(RESULTS / f"heads_{s.name}_s{seed}")
    logger.info(f"heads_{s.name}_s{seed}: {len(z['rows'])} frames, {time.time() - t0:.0f}s")


def finish(
    spec: SpecOption,
    submission: Annotated[Path, typer.Option(help="DIII-D submission .npz to finish")],
    out: Annotated[Path, typer.Option(help="where the finished .npz goes")],
    config: str = "diii_d_public_test",
    cache_root: Path = CACHE_ROOT,
    workers: int = 14,
) -> None:
    """Finish an existing DIII-D submission .npz (shot order = the config's cache) and write it with its scalars."""
    fr = Frames.load(cache_root / config)
    z = np.load(submission)
    shots = range(fr.n_shots)
    psi = np.concatenate([z[f"shot_{i:04d}_psirz"] for i in shots]).astype(np.float32)
    scal = np.stack([np.concatenate([z[f"shot_{i:04d}_{k}"] for i in shots]) for k in ("q95", "betaN")], 1)
    Submission.write_npz(fr, finish_frames(fr, psi, spec, workers), scal, out)


COMMANDS = [finish_targets, train_heads, finish]
