"""The linearised consistency loss (docs/specs/2026-09-30-consistency-loss-design.md, negative): precompute the
gradients, then fine-tune a saved model with the loss added."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger

from fusioneq.consistency import ConsistencyGradients, fine_tune
from fusioneq.evaluation import HeldOut
from fusioneq.model import NeuralPredictor
from fusioneq.settings import CACHE_ROOT, RESULTS
from fusioneq.tracking import Run


def prepare_consistency(
    base: str = "v5_loss_mix_i03",
    n_frames: int = 60_000,
    n_error: int = 20_000,
    workers: int = 15,
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Consistency-loss gradients (C2) on a random training subset, saved as cons_grad_<base>.npz."""
    ev = HeldOut.load(cache_root, holdout, score_shots)
    fr, model = ev.fr, NeuralPredictor.load(RESULTS / base)
    rng = np.random.default_rng(0)
    candidates = np.nonzero(ev.train)[0]
    rows = np.sort(rng.choice(candidates, n_frames, replace=False))
    sample = np.sort(rng.choice(candidates, n_error, replace=False))
    mask = np.zeros(len(fr.shot_index), bool)
    mask[sample] = True
    pred, _ = model.predict(fr, mask)
    B, c = model.pca.components_, model.c_std
    error = ((pred - np.asarray(fr.psi[sample], np.float32)).reshape(len(sample), -1) @ B.T) / c
    grads = ConsistencyGradients.compute(np.asarray(fr.psi[rows], np.float32), rows, B, c, error, workers)
    out = ConsistencyGradients.path(cache_root, base)
    grads.save(out)
    logger.info(f"{out}: {len(grads.rows)}/{n_frames} frames, coefficients {grads.index.tolist()}, var {grads.var}")


def finetune_consistency(
    base: str = "v5_loss_mix_i03",
    weight: float = 1.0,
    epochs: int = 10,
    lr: float = 1e-4,
    tag: str = "cons_ft",
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Fine-tune `base` with the consistency loss (C3), score and save as results/<tag>."""
    t0, ev = time.time(), HeldOut.load(cache_root, holdout, score_shots)
    cons = ConsistencyGradients.load(ConsistencyGradients.path(cache_root, base))
    model = NeuralPredictor.load(RESULTS / base)
    model.cfg.epochs, model.cfg.lr = epochs, lr
    record = {"model": "mlp", **model.cfg.__dict__, "base": base, "cons_weight": weight, "cons_frames": len(cons.rows)}
    run = Run(tag, {**record, "holdout": holdout})
    fine_tune(model, ev.fr, ev.train, cons, weight, ev.subset, run)
    ev.record(tag, model, record, run, t0)


COMMANDS = [prepare_consistency, finetune_consistency]
