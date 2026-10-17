"""Memory-safe access to the flux target: chunked construction, subsampled PCA fit, chunked transform. The target is
either the plasma residual (psi - psi_vac) or, with `normalize`, the frame's shape: psi centred and scaled by its
mean and spread inside the scorer's envelope, whose (mean, log spread) the network predicts separately."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import numpy as np
from sklearn.decomposition import PCA

from fusioneq.cache import Frames
from fusioneq.score import MASKS

CHUNK = 50_000


@cache
def envelope() -> np.ndarray:
    """(4225,) the scorer's DIII-D envelope, where the boundary and the consistency scalars are derived."""
    return np.load(MASKS["DIII-D"])["mask_coarse"].astype(bool).ravel()


def shape_stats(flat: np.ndarray) -> np.ndarray:
    """(N, 2) mean and log standard deviation of each flat (N, 4225) frame inside the envelope."""
    inside = flat[:, envelope()].astype(np.float64)
    return np.stack([inside.mean(1), np.log(inside.std(1) + 1e-12)], 1).astype(np.float32)


def unit_shape(flat: np.ndarray, stats: np.ndarray | None = None) -> np.ndarray:
    """Flat (N, 4225) frames centred and scaled by their envelope mean and spread (`stats` from shape_stats)."""
    st = shape_stats(flat) if stats is None else stats
    return (flat - st[:, :1]) / np.exp(st[:, 1:])


def match_envelope(psi: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """(N, 65, 65) `psi`'s shape moved onto per-frame envelope `stats` (mean, log spread, as from `shape_stats`)."""
    a = shape_stats(psi.reshape(len(psi), -1))
    out = psi - a[:, 0][:, None, None]
    out *= np.exp(stats[:, 1] - a[:, 1])[:, None, None]
    out += stats[:, 0][:, None, None]
    return out


def residual_chunks(
    fr: Frames, rows: np.ndarray, use_vacuum: bool, chunk: int = CHUNK, normalize: bool = False
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (row_indices, flat float32 target) for consecutive slices of `rows` (sorted frame indices)."""
    for i in range(0, len(rows), chunk):
        idx = rows[i : i + chunk]
        t = fr.psi[idx].astype(np.float32).reshape(len(idx), -1)
        if normalize:
            t = unit_shape(t)
        elif use_vacuum:
            t = t - fr.psi_vac[idx].astype(np.float32).reshape(len(idx), -1)
        yield idx, t


def fit_pca(
    fr: Frames,
    rows: np.ndarray,
    n_components: int,
    use_vacuum: bool,
    n_fit: int = 150_000,
    seed: int = 0,
    normalize: bool = False,
) -> PCA:
    """Fit PCA on a random subsample of frames (a 150k x 4225 float32 matrix is 2.5 GB)."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(rows, min(n_fit, len(rows)), replace=False))
    flat = np.concatenate([t for _, t in residual_chunks(fr, pick, use_vacuum, normalize=normalize)])
    return PCA(n_components, random_state=seed, svd_solver="randomized").fit(flat)


def transform_all(pca: PCA, fr: Frames, rows: np.ndarray, use_vacuum: bool, normalize: bool = False) -> np.ndarray:
    """PCA coefficients for every row, never holding more than one chunk of targets."""
    out = np.empty((len(rows), pca.n_components_), np.float32)
    pos = 0
    for _, t in residual_chunks(fr, rows, use_vacuum, normalize=normalize):
        out[pos : pos + len(t)] = pca.transform(t).astype(np.float32)
        pos += len(t)
    return out


def transform_shape(pca: PCA, fr: Frames, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(PCA coefficients of the normalized shape, (N, 2) shape_stats) for every row, reading each frame once."""
    coef = np.empty((len(rows), pca.n_components_), np.float32)
    stats = np.empty((len(rows), 2), np.float32)
    for i in range(0, len(rows), CHUNK):
        idx = rows[i : i + CHUNK]
        t = fr.psi[idx].astype(np.float32).reshape(len(idx), -1)
        st = shape_stats(t)
        coef[i : i + len(idx)] = pca.transform(unit_shape(t, st)).astype(np.float32)
        stats[i : i + len(idx)] = st
    return coef, stats


def reconstruct(
    pca: PCA, coef: np.ndarray, fr: Frames, rows: np.ndarray, use_vacuum: bool, stats: np.ndarray | None = None
) -> np.ndarray:
    """(N, 65, 65) psi from coefficients, chunk by chunk: un-normalised by `stats` (mean, log spread) when given,
    else with the vacuum term added back."""
    out = np.empty((len(rows), 65, 65), np.float32)
    for i in range(0, len(rows), CHUNK):
        sl = slice(i, i + CHUNK)
        psi = pca.inverse_transform(coef[sl]).reshape(-1, 65, 65).astype(np.float32)
        if stats is not None:
            psi = psi * np.exp(stats[sl, 1])[:, None, None] + stats[sl, 0][:, None, None]
        elif use_vacuum:
            psi += fr.psi_vac[rows[sl]].astype(np.float32)
        out[sl] = psi
    return out


def r2_psi(fr: Frames, rows: np.ndarray, psi_pred: np.ndarray) -> float:
    """Pooled R2 of a prediction against the cached ground truth, chunked."""
    ss_res = ss_tot = 0.0
    s = n = 0.0
    for i in range(0, len(rows), CHUNK):
        g = fr.psi[rows[i : i + CHUNK]].astype(np.float32)
        s += float(g.sum())
        n += g.size
    mean = s / n
    for i in range(0, len(rows), CHUNK):
        g = fr.psi[rows[i : i + CHUNK]].astype(np.float32)
        ss_res += float(((psi_pred[i : i + CHUNK] - g) ** 2).sum())
        ss_tot += float(((g - mean) ** 2).sum())
    return 1 - ss_res / ss_tot
