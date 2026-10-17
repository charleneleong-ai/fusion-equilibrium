"""Grid shape model (fusioneq.grid): train it on the cached DIII-D frames."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from loguru import logger

from fusioneq.evaluation import HeldOut
from fusioneq.grid import GridConfig, GridPredictor
from fusioneq.settings import CACHE_ROOT, RESULTS
from fusioneq.tracking import Run
from fusioneq.viz import pick_frames, shape_triptych


def train_grid(
    tag: str = "grid",
    arch: str = "unet",
    channels: int = 32,
    modes: int = 16,
    layers: int = 4,
    context: int = 4,
    epochs: int = 10,
    batch: int = 256,
    stride: int = 2,
    lr: float = 1e-3,
    env_weight: float = 4.0,
    gs_weight: float = 0.0,
    seed: int = 0,
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Fit the grid shape model (--arch unet, or fno: --channels is then its width, with --modes and --layers) on
    every stride-th training frame, keeping the epoch with the lowest envelope loss on the never-scored validation
    shots; save as results/<tag>."""
    ev = HeldOut.load(cache_root, holdout, score_shots)
    cfg = GridConfig(
        arch=arch,
        channels=channels,
        modes=modes,
        layers=layers,
        context=context,
        epochs=epochs,
        batch=batch,
        stride=stride,
        lr=lr,
        env_weight=env_weight,
        gs_weight=gs_weight,
        seed=seed,
    )
    run = Run(tag, {"model": "grid", **asdict(cfg), "holdout": holdout})
    m = GridPredictor(cfg).fit(ev.fr, ev.train, ev.validation, run, checkpoint=RESULTS / tag)
    m.save(RESULTS / tag)
    logger.info(f"{tag}: validation envelope loss {m.val_loss:.5f}")
    rows = np.nonzero(ev.validation)[0]
    picked = rows[pick_frames(ev.fr, rows, 4)]
    shapes = m.predict_shape(ev.fr, np.isin(np.arange(len(ev.fr.shot_index)), picked))
    run.log_images({"figures/shape": shape_triptych(ev.fr, picked, shapes, RESULTS / "figures", tag)})
    run.summary({"val_envelope_loss": m.val_loss})
    run.finish()


COMMANDS = [train_grid]
