"""Figures for a scored model: flux-map and flux-shape triptychs on held-out frames, and scalar scatter plots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from fusioneq.cache import Frames  # noqa: E402
from fusioneq.targets import envelope, unit_shape  # noqa: E402


def pick_frames(fr: Frames, rows: np.ndarray, n: int) -> np.ndarray:
    """One mid-shot frame from each of the first n held-out shots (positions into `rows`)."""
    picks = []
    for sid in np.unique(fr.shot_index[rows])[:n]:
        where = np.nonzero(fr.shot_index[rows] == sid)[0]
        picks.append(where[len(where) // 2])
    return np.asarray(picks)


def save_figure(fig: Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def triptych(
    fr: Frames,
    rows: np.ndarray,
    truth: list[np.ndarray],
    pred: np.ndarray,
    names: tuple[str, str],
    error_title: Callable[[np.ndarray], str],
    outline: np.ndarray | None = None,
) -> Figure:
    """Truth and prediction per frame on the truth's contour levels, and the |error| map; `outline` is drawn in red."""
    R, Z = fr.grid_R, fr.grid_Z
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for k, row in enumerate(rows):
        levels = np.linspace(truth[k].min(), truth[k].max(), 14)
        for ax, field, name in ((axes[k, 0], truth[k], names[0]), (axes[k, 1], pred[k], names[1])):
            ax.contourf(R, Z, field, levels=levels, cmap="viridis")
            ax.contour(R, Z, field, levels=levels, colors="k", linewidths=0.4)
            ax.set_title(f"{name}, shot {fr.shot_index[row]} t={fr.efit_times[row]:.0f} ms", fontsize=9)
        err = np.abs(pred[k] - truth[k])
        im = axes[k, 2].pcolormesh(R, Z, err, cmap="magma", shading="auto")
        axes[k, 2].set_title(error_title(err), fontsize=9)
        fig.colorbar(im, ax=axes[k, 2], shrink=0.8)
        for ax in axes[k]:
            if outline is not None:
                ax.contour(R, Z, outline, levels=[0.5], colors="r", linewidths=1.0)
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
    return fig


def flux_triptych(fr: Frames, rows: np.ndarray, psi_pred: np.ndarray, out: Path, tag: str, n: int = 4) -> Path:
    picks = pick_frames(fr, rows, n)
    truth = [fr.psi[rows[i]].astype(np.float32) for i in picks]
    fig = triptych(
        fr,
        rows[picks],
        truth,
        psi_pred[picks],
        ("EFIT psi", "predicted psi"),
        lambda err: f"|error|, max {err.max():.3f} Wb/rad",
    )
    fig.suptitle(f"{tag}: held-out DIII-D frames")
    return save_figure(fig, out, f"{tag}_flux.png")


def shape_triptych(fr: Frames, rows: np.ndarray, shape_pred: np.ndarray, out: Path, tag: str) -> Path:
    """EFIT and predicted flux shape (envelope units) for `rows`, and the error; the scorer's envelope outlined in red,
    where the boundary and the consistency scalars are derived."""
    env = envelope().reshape(65, 65).astype(float)
    truth = [unit_shape(fr.psi[row].astype(np.float32).reshape(1, -1)).reshape(65, 65) for row in rows]
    fig = triptych(
        fr,
        rows,
        truth,
        shape_pred,
        ("EFIT shape", "predicted shape"),
        lambda err: f"|error| (envelope units), max {err[env > 0].max():.3f} inside",
        outline=env,
    )
    fig.suptitle(f"{tag}: flux shape on held-out DIII-D frames (red: scorer envelope)")
    return save_figure(fig, out, f"{tag}_shape.png")


def scalar_scatter(fr: Frames, rows: np.ndarray, scal_pred: np.ndarray, out: Path, tag: str) -> Path:
    truth = fr.scalars[rows][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for j, (ax, name) in enumerate(zip(axes, ("q95", "betaN"), strict=True)):
        ok = np.isfinite(truth[:, j])
        ax.scatter(truth[ok, j], scal_pred[ok, j], s=3, alpha=0.3)
        lo, hi = np.nanpercentile(truth[ok, j], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"EFIT {name}")
        ax.set_ylabel(f"predicted {name}")
    fig.suptitle(f"{tag}: submitted scalars on held-out frames")
    return save_figure(fig, out, f"{tag}_scalars.png")


def render_all(
    fr: Frames, rows: np.ndarray, psi_pred: np.ndarray, scal_pred: np.ndarray, out: Path, tag: str
) -> dict[str, Path]:
    return {
        "figures/flux": flux_triptych(fr, rows, psi_pred, out, tag),
        "figures/scalars": scalar_scatter(fr, rows, scal_pred, out, tag),
    }
