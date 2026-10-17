"""Machine-agnostic plasma flux: the measured plasma current spread over an ellipse located from Thomson profiles.

psi_total = psi_vac (coils) + psi_plasma, where psi_plasma is the flux of a parabolic current density
J ~ (1 - rho^2)^alpha inside rho^2 = (x + delta y^2)^2 + y^2 <= 1, x = (R - R0)/a, y = (Z - Z0)/(kappa a), normalised
to Ip. delta (upper half delta_top, lower half delta_bot) is the first-order D shape R = R0 + a cos(t + delta sin t), so
delta = 0 is the plain ellipse. R0 and a come from where the Thomson density profile sits above a fraction of its peak,
so no equilibrium labels of the target machine are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlasmaShape:
    kappa: float = 1.8
    alpha: float = 1.0
    delta_top: float = 0.0
    delta_bot: float = 0.0


def density_extent(R: np.ndarray, ne: np.ndarray, frac: float = 0.25) -> tuple[float, float] | None:
    """(R_in, R_out) where one frame's density profile exceeds `frac` of its peak; None when the profile is empty."""
    ok = np.isfinite(ne) & np.isfinite(R) & (ne > 0)
    if ok.sum() < 3:
        return None
    r, n = R[ok], ne[ok]
    order = np.argsort(r)
    r, n = r[order], n[order]
    inside = n >= frac * n.max()
    return float(r[inside].min()), float(r[inside].max())


def ellipse_rho2(R: np.ndarray, Z: np.ndarray, r0: Any, a: Any, z0: Any, kappa: Any) -> np.ndarray:
    """Squared normalised radius on elliptical surfaces centred at (r0, z0), half-width a, elongation kappa."""
    return ((R - r0) / a) ** 2 + ((Z - z0) / (kappa * a)) ** 2


def shaped_rho2(R: np.ndarray, Z: np.ndarray, r0: float, a: float, z0: float, shape: PlasmaShape) -> np.ndarray:
    """Squared normalised radius on D-shaped surfaces; the ellipse's when both triangularities are 0."""
    y = (Z - z0) / (shape.kappa * a)
    delta = np.where(y >= 0, shape.delta_top, shape.delta_bot)
    return ((R - r0) / a + delta * y**2) ** 2 + y**2


def current_density(
    grid_R: np.ndarray, grid_Z: np.ndarray, r0: float, a: float, z0: float, shape: PlasmaShape
) -> np.ndarray:
    """(65, 65) weights summing to 1 over the plasma ellipse centred at (r0, z0), rows = Z, cols = R."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    rho2 = shaped_rho2(RR, ZZ, r0, a, z0, shape)
    j = np.clip(1.0 - rho2, 0.0, None) ** shape.alpha
    total = j.sum()
    return j / total if total > 0 else j
