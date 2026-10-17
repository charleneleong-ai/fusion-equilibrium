"""Linearised consistency loss (docs/specs/2026-09-30-consistency-loss-design.md): per-frame gradients of the scorer's
integral consistency scalars w.r.t. the network's scaled PCA coefficients, by central differences through the
scorer's own `derive_frame`, and a fine-tune that adds the linearised error to the loss. Only volume and li linearise
(C1), and fine-tuning with them overfits (C3, negative); kept to reproduce that result.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fusioneq.cache import Frames
from fusioneq.model import NeuralPredictor, context_rows
from fusioneq.score import Scorer, derive_frame
from fusioneq.targets import transform_all
from fusioneq.tracking import Run

TARGETS = ("volume", "li")
TOP_K = 16  # 97.7% of v5_loss_mix_i03's held-out error energy; more coefficients only add difference noise
BATCH = 512  # frames with gradients per step, alongside the usual batch
SCORER = Scorer("DIII-D")
STEPS: np.ndarray | None = None  # (TOP_K, 65, 65) psi steps, set once per worker process


def targets(psi: np.ndarray) -> np.ndarray:
    """(len(TARGETS),) the scorer's volume and li of one (65, 65) frame, NaN where its derivation fails."""
    v = derive_frame(np.asarray(psi, np.float64), SCORER.R, SCORER.Z, "DIII-D", SCORER.mask_coarse, SCORER.mask_f)
    return np.array([v[k] for k in TARGETS])


def set_steps(steps: np.ndarray) -> None:
    global STEPS
    STEPS = steps


def frame_gradients(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(f(label), g): g[k] is the central difference of the targets along STEPS[k]."""
    return targets(label), np.stack([(targets(label + s) - targets(label - s)) / 2 for s in STEPS])


@dataclass
class ConsistencyGradients:
    """Gradients for a subset of training rows: g[n, k, j] = change of target j per unit step along coefficient
    index[k]; a coefficient error d (scaled units) moves target j by sum_k g[n, k, j] * d[index[k]] / step[k]."""

    rows: np.ndarray  # (N,) training rows
    index: np.ndarray  # (TOP_K,) coefficient indices
    step: np.ndarray  # (TOP_K,) step per coefficient, scaled units
    g: np.ndarray  # (N, TOP_K, len(TARGETS))
    var: np.ndarray  # (len(TARGETS),) variance of each target over the subset: the scorer's R2 denominator

    @classmethod
    def compute(
        cls,
        label: np.ndarray,
        rows: np.ndarray,
        components: np.ndarray,
        c_std: np.ndarray,
        error: np.ndarray,
        workers: int,
    ) -> ConsistencyGradients:
        """label (N, 65, 65); error (M, n_components) a sample of the model's scaled-coefficient errors, which sets
        which coefficients matter (energy) and the step on each (their spread)."""
        energy = ((error * c_std) ** 2).mean(0)
        index = np.sort(np.argsort(energy)[::-1][:TOP_K])
        step = error[:, index].std(0)
        steps = ((step * c_std[index])[:, None] * components[index]).reshape(-1, *label.shape[1:])
        with ProcessPoolExecutor(workers, initializer=set_steps, initargs=(steps,)) as ex:
            out = list(ex.map(frame_gradients, label, chunksize=64))
        f0, g = np.stack([o[0] for o in out]), np.stack([o[1] for o in out]).astype(np.float32)
        ok = np.isfinite(f0).all(1) & np.isfinite(g).all((1, 2))
        return cls(rows[ok], index, step.astype(np.float32), g[ok], np.nanvar(f0[ok], 0).astype(np.float32))

    @staticmethod
    def path(cache_root: Path, base: str) -> Path:
        return cache_root / "diii_d_train" / f"cons_grad_{base}.npz"

    def save(self, path: Path) -> None:
        np.savez(path, rows=self.rows, index=self.index, step=self.step, g=self.g, var=self.var)

    @classmethod
    def load(cls, path: Path) -> ConsistencyGradients:
        z = np.load(path)
        return cls(z["rows"], z["index"], z["step"], z["g"], z["var"])


def consistency_loss(
    d: torch.Tensor, g: torch.Tensor, index: torch.Tensor, step: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """Linearised consistency SSE over the targets, each over its variance: sum_j mean_n (g_nj . d_n)^2 / var_j,
    for d (B, n_components) scaled-coefficient errors and g (B, TOP_K, len(TARGETS))."""
    delta = torch.einsum("bkj,bk->bj", g, d[:, index] / step)
    return (delta**2 / var).mean(0).sum()


def fine_tune(
    model: NeuralPredictor,
    fr: Frames,
    train: np.ndarray,
    cons: ConsistencyGradients,
    weight: float,
    val: np.ndarray | None = None,
    run: Run | None = None,
) -> NeuralPredictor:
    """Continue training a loaded model (its own normalisation and PCA; cfg.epochs and cfg.lr set the schedule) with
    `weight` x the linearised consistency loss on a batch of the frames `cons` covers, added to every step."""
    if model.cfg.temporal_layers or model.cfg.normalize:
        raise NotImplementedError("the consistency fine-tune supports the per-frame residual-PCA MLP only")
    torch.manual_seed(model.cfg.seed)
    rows = np.nonzero(train)[0]
    at = np.searchsorted(rows, cons.rows)
    if not np.array_equal(rows[at], cons.rows):
        raise ValueError("consistency gradients cover frames outside the training rows")
    coef = transform_all(model.pca, fr, rows, model.cfg.use_vacuum)
    X, y, s_ok = model._training_arrays(fr, rows, coef, context_rows(fr, rows, model.cfg))
    at, g, index, step, var = (
        torch.as_tensor(a, device=model.device) for a in (at, cons.g, cons.index, cons.step, cons.var)
    )
    nc = model.cfg.n_components

    def extra_loss(m: NeuralPredictor, Xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        j = torch.randint(len(at), (BATCH,), device=m.device)  # frames with gradients; at[j] their training rows
        d = m.forward(Xt[at[j]])[:, :nc] - yt[at[j], :nc]
        return weight * consistency_loss(d, g[j], index, step, var)

    model._train_loop(X, y, s_ok, fr, val, run, extra_loss)
    return model
