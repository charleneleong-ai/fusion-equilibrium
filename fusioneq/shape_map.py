"""Per-frame plasma shape from the coil vacuum field: a low-capacity, dimensionless map fitted on DIII-D only.

The vacuum flux is expanded to third order about the plasma centre (R0, 0) in units of the minor radius a,
psi = sum c_ij x^i y^j with x = (R - R0)/a, y = Z/a. The map reads only ratios of these moments, which mean the same
thing on any tokamak: the field-line tilt c01/c10 (radial over vertical field: what pushes the plasma up or down),
the decay index n = -(R/Bz) dBz/dR (elongation), the hexapole terms (triangularity) and the vertical field over the
plasma's own hoop field (Shafranov's Lambda). Targets are the scorer's derived Z_axis, kappa, tri_top and tri_bot of
DIII-D EFIT flux, where labels are allowed. No MAST equilibrium data is read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fusioneq.greens import MU0

BASIS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))
FEATURES = ("tilt", "decay_index", "quad_tilt", "hex_30", "hex_21", "hex_12", "hex_03", "lambda_v")
TARGETS = ("Z_axis", "kappa", "tri_top", "tri_bot")
CONS_INDEX = {"R_axis": 0, "Z_axis": 1, "kappa": 2, "tri_top": 3, "tri_bot": 4, "volume": 5, "li": 6}
CLIP_QUANTILES = (0.5, 99.5)  # features and predictions stay inside DIII-D's training range
DELTA_MAX = 0.5  # the axis stays within half a minor radius of the boundary centre (a judgement, not a fit)


def stencil(n_x: int = 11, n_y: int = 15, y_max: float = 1.5) -> np.ndarray:
    """(P, 2) normalised (x, y) sample points inside the ellipse x^2 + (y / y_max)^2 <= 1."""
    x, y = np.meshgrid(np.linspace(-1, 1, n_x), np.linspace(-y_max, y_max, n_y))
    keep = x**2 + (y / y_max) ** 2 <= 1 + 1e-9
    return np.stack([x[keep], y[keep]], 1)


def bilinear(psi: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """(N, P) psi[n] (rows = Z, cols = R, uniform grid) sampled at points (R[n, p], Z[n, p]), clamped to the grid."""
    fr = np.clip((R - grid_R[0]) / (grid_R[1] - grid_R[0]), 0, len(grid_R) - 1.000001)
    fz = np.clip((Z - grid_Z[0]) / (grid_Z[1] - grid_Z[0]), 0, len(grid_Z) - 1.000001)
    ir, iz = fr.astype(int), fz.astype(int)
    tr, tz = fr - ir, fz - iz
    n = np.arange(len(psi))[:, None]

    def at(dz: int, dr: int) -> np.ndarray:
        return psi[n, iz + dz, ir + dr]

    return (at(0, 0) * (1 - tr) + at(0, 1) * tr) * (1 - tz) + (at(1, 0) * (1 - tr) + at(1, 1) * tr) * tz


def moments(psi: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray, r0: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(N, len(BASIS)) least-squares polynomial coefficients c_ij of each frame about (r0, 0) in units of a."""
    pts = stencil()
    A = np.stack([pts[:, 0] ** i * pts[:, 1] ** j for i, j in BASIS], 1)
    R = r0[:, None] + a[:, None] * pts[None, :, 0]
    Z = a[:, None] * pts[None, :, 1]
    samples = bilinear(np.asarray(psi, np.float64), grid_R, grid_Z, R, Z)
    return samples @ np.linalg.pinv(A).T


def features(
    psi_vac: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray, r0: np.ndarray, a: np.ndarray, ip_ka: np.ndarray
) -> np.ndarray:
    """(N, len(FEATURES)) dimensionless vacuum-field descriptors; every one is invariant under psi -> -psi."""
    r0, a = np.asarray(r0, np.float64), np.asarray(a, np.float64)
    c = moments(psi_vac, grid_R, grid_Z, r0, a)
    k = {ij: c[:, n] for n, ij in enumerate(BASIS)}
    c10 = k[(1, 0)]
    c10 = np.where(np.abs(c10) > 1e-12, c10, 1e-12)
    ip_a = np.maximum(np.abs(np.asarray(ip_ka, np.float64)) * 1e3, 1e3)
    # Bz = (1/R) dpsi/dR = c10 / (a R0); Shafranov: Bz = mu0 Ip / (4 pi R0) * (ln(8 R0/a) + Lambda - 1/2)
    lambda_v = np.abs(c10) * 4 * np.pi / (a * MU0 * ip_a) - np.log(8 * r0 / a) + 0.5
    cols = [
        k[(0, 1)] / c10,
        1 - 2 * (r0 / a) * k[(2, 0)] / c10,
        k[(1, 1)] / c10,
        k[(3, 0)] / c10,
        k[(2, 1)] / c10,
        k[(1, 2)] / c10,
        k[(0, 3)] / c10,
        lambda_v,
    ]
    return np.stack(cols, 1)


@dataclass(frozen=True)
class ShapeMap:
    """Ridge regression per target on standardised, range-clipped features; predictions clipped to the DIII-D
    training range of each target."""

    lam: float
    x_lo: list[float]
    x_hi: list[float]
    mu: list[float]
    sd: list[float]
    coef: list[list[float]]  # (T, F)
    intercept: list[float]
    y_lo: list[float]
    y_hi: list[float]
    targets: tuple[str, ...] = TARGETS

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> ShapeMap:
        ok = np.isfinite(X).all(1)
        lo, hi = np.percentile(X[ok], CLIP_QUANTILES, axis=0)
        Xc = np.clip(X, lo, hi)
        mu, sd = Xc[ok].mean(0), Xc[ok].std(0) + 1e-12
        coef, icpt = [], []
        for j in range(Y.shape[1]):
            rows = ok & np.isfinite(Y[:, j])
            Zs, y = (Xc[rows] - mu) / sd, Y[rows, j]
            w = np.linalg.solve(Zs.T @ Zs + lam * len(Zs) * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
            coef.append(w.tolist())
            icpt.append(float(y.mean()))
        y_lo, y_hi = np.nanpercentile(Y, CLIP_QUANTILES, axis=0)
        return cls(lam, lo.tolist(), hi.tolist(), mu.tolist(), sd.tolist(), coef, icpt, y_lo.tolist(), y_hi.tolist())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, T); non-finite feature rows get each target's training mean."""
        Zs = (np.clip(X, self.x_lo, self.x_hi) - np.array(self.mu)) / np.array(self.sd)
        out = np.array(self.intercept) + Zs @ np.array(self.coef).T
        out = np.where(np.isfinite(out), out, np.array(self.intercept))
        return np.clip(out, self.y_lo, self.y_hi)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> ShapeMap:
        d = json.loads(path.read_text())
        return cls(**{**d, "targets": tuple(d["targets"])})


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Pooled R2 over frames where both are finite (the scorer's per-scalar form, before its clip at 0)."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))


def placement_shape(geom: np.ndarray, pred: np.ndarray, use_tri: bool = True) -> np.ndarray:
    """(N, 4) R0, a, kappa, delta for the q95 formula with no boundary extraction: the plasma placement rows (R0, a)
    and the ShapeMap's predicted (Z_axis, kappa, tri_top, tri_bot) rows, delta the mean of the two triangularities."""
    t = {name: pred[:, j] for j, name in enumerate(TARGETS)}
    delta = (t["tri_top"] + t["tri_bot"]) / 2 if use_tri else np.zeros(len(pred))
    return np.c_[geom[:, 0], geom[:, 1], t["kappa"], delta]


@dataclass(frozen=True)
class ShiftModel:
    """Shafranov shift of the magnetic axis outward from the boundary centre, Delta / a = c0 + c1 * x with
    x = eps (beta_p + li / 2) (scalars.shafranov_drive), least squares on DIII-D labels; predictions clipped to
    [0, DELTA_MAX] of a. The physics form, not a clipped regression, so it extrapolates to MAST's larger eps."""

    c0: float
    c1: float

    @classmethod
    def fit(cls, x: np.ndarray, delta_over_a: np.ndarray) -> ShiftModel:
        ok = np.isfinite(x) & np.isfinite(delta_over_a)
        c1, c0 = np.polyfit(x[ok], delta_over_a[ok], 1)
        return cls(float(c0), float(c1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """(N,) Delta / a; frames with no drive (missing pressure) take the drive's median."""
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        return np.clip(self.c0 + self.c1 * x, 0.0, DELTA_MAX)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> ShiftModel:
        return cls(**json.loads(path.read_text()))
