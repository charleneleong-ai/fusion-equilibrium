"""Reshape the core flux to a target internal inductance while keeping the boundary (the li lever).

Inside the flux's own LCFS (the scorer's extractor, no envelope), with x = (phi - phi_b) / (phi_axis - phi_b) the
normalised flux (0 on the LCFS, 1 at the axis), the remap is

    phi -> phi + c * (phi_axis - phi_b) * x^2

It leaves the LCFS level and every flux surface's nesting unchanged (monotone for c > -1/2), keeps |grad phi| continuous
across the boundary (the slope at x = 0 is untouched, so the plasma current read from the boundary stays the same),
and keeps the axis cell. c < 0 flattens the current profile and lowers li; c > 0 peaks it. The scorer's li(2) of the
remapped flux is tabulated over C_GRID per frame, and c is interpolated to the frame's target li.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from fusioneq.score import extract_lcfs_with_sign

C_GRID = tuple(np.linspace(-0.45, 1.5, 40).tolist())


@dataclass(frozen=True)
class CoreShaper:
    """Per-frame li(c) curves and the remap; axis_sign puts the axis at a maximum of axis_sign * psi."""

    R: np.ndarray
    Z: np.ndarray
    machine: str
    axis_sign: float
    c_grid: tuple[float, ...] = C_GRID

    def bump(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(bump, contour): the remap direction (phi_axis - phi_b) * x^2 inside the LCFS (zero elsewhere, and
        everywhere when no LCFS closes), in psi's own sign, and the LCFS polygon or None."""
        p = np.asarray(psi, np.float64)
        c, _ = extract_lcfs_with_sign(p, self.R, self.Z, self.machine, axis_sign=self.axis_sign)
        if c is None:
            return np.zeros_like(p), None
        phi = self.axis_sign * p
        inside = points_in_polygon(self.R, self.Z, c)
        phi_b = float(np.mean(bilinear(phi, self.R, self.Z, c[:, 0], c[:, 1])))
        depth = float(phi[inside].max()) - phi_b if inside.any() else 0.0
        if depth <= 0:
            return np.zeros_like(p), c
        x = np.clip((phi - phi_b) / depth, 0.0, 1.0) * inside
        return self.axis_sign * depth * x**2, c

    def curve(self, psi: np.ndarray, bump: np.ndarray, contour: np.ndarray | None) -> np.ndarray:
        """(len(c_grid),) the scorer's li(2) of psi + c * bump over the grid, vectorised over c (same sums as
        derive.internal_inductance); NaN without a contour."""
        if contour is None:
            return np.full(len(self.c_grid), np.nan)
        inside = points_in_polygon(self.R, self.Z, contour)
        if inside.sum() < 4:
            return np.full(len(self.c_grid), np.nan)
        dz_p, dr_p = np.gradient(np.asarray(psi, np.float64), self.Z, self.R)
        dz_b, dr_b = np.gradient(np.asarray(bump, np.float64), self.Z, self.R)
        c = np.asarray(self.c_grid)[:, None, None]
        bp = np.sqrt((dr_p + c * dr_b) ** 2 + (dz_p + c * dz_b) ** 2) / self.R[None, None, :]
        bp = np.where(np.isfinite(bp), bp, 0.0)
        dv = 2 * np.pi * self.R[None, :] * np.mean(np.diff(self.R)) * np.mean(np.diff(self.Z))
        dv = np.broadcast_to(dv, inside.shape)[inside]
        bp2_v = (bp[:, inside] ** 2 * dv).sum(1) / dv.sum()
        on_c = np.stack([bilinear(b, self.R, self.Z, contour[:, 0], contour[:, 1]) for b in bp])
        seg = np.sqrt((np.diff(np.vstack([contour, contour[:1]]), axis=0) ** 2).sum(1))
        mu0_ip = (0.5 * (on_c + np.roll(on_c, -1, axis=1)) * seg).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mu0_ip > 0, bp2_v / (mu0_ip / seg.sum()) ** 2, np.nan)

    def rows(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(bumps (N, nz, nr) float32, curves (N, len(c_grid)))."""
        bumps, curves = np.zeros(psi.shape, np.float32), np.full((len(psi), len(self.c_grid)), np.nan)
        for k, p in enumerate(np.asarray(psi, np.float64)):
            b, c = self.bump(p)
            bumps[k], curves[k] = b, self.curve(p, b, c)
        return bumps, curves

    def many(self, psi: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray]:
        chunks = np.array_split(np.arange(len(psi)), max(1, min(len(psi), workers * 4)))
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(self.rows, [np.asarray(psi[c]) for c in chunks]))
        return np.concatenate([b for b, _ in parts]), np.concatenate([c for _, c in parts])

    def coefficient(self, curves: np.ndarray, target: np.ndarray) -> np.ndarray:
        """(N,) c whose tabulated li is nearest the target (interpolated, clipped to the grid); 0 where the curve or the
        target is missing."""
        out = np.zeros(len(curves))
        for k, (li, t) in enumerate(zip(curves, target, strict=True)):
            if not (np.isfinite(t) and np.isfinite(li).all()):
                continue
            if np.all(np.diff(li) > 0):
                out[k] = np.interp(t, li, self.c_grid)
            else:
                out[k] = self.c_grid[int(np.argmin(np.abs(li - t)))]
        return out


def shrink_targets(li: np.ndarray, level: float, k: float, center: float | None = None) -> np.ndarray:
    """Target li = level + k (li - center): our own li's frame-to-frame variation shrunk by k about `center` (the
    median of `li` by default) and moved to a level set from outside the predictions."""
    center = float(np.nanmedian(li)) if center is None else center
    return level + k * (li - center)


def shot_smooth(v: np.ndarray, shot_index: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling nan-median of v over `window` frames within each shot (frames contiguous per shot); window <= 1
    returns v."""
    if window <= 1:
        return np.asarray(v, np.float64)
    out, h = np.full(len(v), np.nan), window // 2
    cuts = np.r_[0, np.flatnonzero(np.diff(np.asarray(shot_index))) + 1, len(v)]
    for s, e in zip(cuts[:-1], cuts[1:], strict=True):
        seg = np.asarray(v[s:e], np.float64)
        for i in range(e - s):
            w = seg[max(0, i - h) : i + h + 1]
            out[s + i] = np.nanmedian(w) if np.isfinite(w).any() else np.nan
    return out


def points_in_polygon(R: np.ndarray, Z: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """(nz, nr) grid cells inside the closed (N, 2) polygon (even-odd rule, as the scorer's li)."""
    RR, ZZ = np.meshgrid(R, Z)
    r1, z1 = contour[:, 0], contour[:, 1]
    r2, z2 = np.roll(r1, -1), np.roll(z1, -1)
    rq, zq = RR.ravel(), ZZ.ravel()
    cross = (z1[:, None] > zq[None]) != (z2[:, None] > zq[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = r1[:, None] + (zq[None] - z1[:, None]) * (r2 - r1)[:, None] / (z2 - z1)[:, None]
    hits = cross & (rq[None] < xint)
    return (hits.sum(0) % 2 == 1).reshape(RR.shape)


def bilinear(field: np.ndarray, R: np.ndarray, Z: np.ndarray, rq: np.ndarray, zq: np.ndarray) -> np.ndarray:
    fr, fz = np.interp(rq, R, np.arange(len(R))), np.interp(zq, Z, np.arange(len(Z)))
    r0, z0 = np.clip(fr.astype(int), 0, len(R) - 2), np.clip(fz.astype(int), 0, len(Z) - 2)
    tr, tz = fr - r0, fz - z0
    return (
        field[z0, r0] * (1 - tr) * (1 - tz)
        + field[z0, r0 + 1] * tr * (1 - tz)
        + field[z0 + 1, r0] * (1 - tr) * tz
        + field[z0 + 1, r0 + 1] * tr * tz
    )
