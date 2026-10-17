"""Vacuum poloidal flux of the shipped coil rectangles: the linear, machine-agnostic part of psi."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi


def filament_flux(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float) -> np.ndarray:
    """psi (Wb/rad) at grid points (R, Z) from a 1 A circular filament at (Rc, Zc)."""
    k2 = 4 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-12, 1 - 1e-12)
    k = np.sqrt(k2)
    return MU0 / (2 * np.pi) * np.sqrt(R * Rc) / k * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2))


def rectangle_flux(
    R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float, width: float, height: float, n_sub: int = 3
) -> np.ndarray:
    """Flux per ampere-turn of a rectangular conductor, as n_sub x n_sub filaments sharing the current."""
    rs = Rc + width * (np.arange(n_sub) + 0.5) / n_sub - width / 2
    zs = Zc + height * (np.arange(n_sub) + 0.5) / n_sub - height / 2
    out = np.zeros_like(R, dtype=np.float64)
    for r in rs:
        for z in zs:
            out += filament_flux(R, Z, float(r), float(z))
    return out / n_sub**2


# Columns shipped as total ampere-turns for the whole coil (DIII-D F-coils, MAST P-coils). Their total is shared by
# the conductor elements joined to the column. Every other column (ECOILA, bcoil, MAST sol/tf/efps) is per turn, so
# each joined element carries the full current.
AMPERE_TURN_COLUMN = re.compile(r"magnetics_(F\d[AB]|p\d[lu]_current)$")


def coil_basis(grid_R: np.ndarray, grid_Z: np.ndarray, coils: dict[str, np.ndarray], columns: list[str]) -> np.ndarray:
    """(C, 65, 65) flux per unit column current: summed over joined elements, averaged for ampere-turn columns."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)  # rows = Z, cols = R
    basis = np.zeros((len(columns), *RR.shape), np.float64)
    join = np.asarray(coils["coil_input_column"]).astype(str)
    for j, col in enumerate(columns):
        for i in np.nonzero(join == col)[0]:
            basis[j] += rectangle_flux(
                RR,
                ZZ,
                float(coils["coil_R"][i]),
                float(coils["coil_Z"][i]),
                float(coils["coil_width"][i]),
                float(coils["coil_height"][i]),
            )
        n = int((join == col).sum())
        if n > 1 and AMPERE_TURN_COLUMN.match(col):
            basis[j] /= n
    return (basis * 1e3).astype(np.float32)  # currents arrive in kA(-turn)


def vacuum_flux(basis: np.ndarray, currents: np.ndarray) -> np.ndarray:
    """(T, 65, 65) = sum_c I_c(t) * basis_c."""
    return np.einsum("tc,chw->thw", currents.astype(np.float32), basis)


def grid_greens(grid_R: np.ndarray, grid_Z: np.ndarray) -> np.ndarray:
    """(N, N) flux at every grid point per ampere at every grid point, for a plasma-current term."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    r, z = RR.ravel(), ZZ.ravel()
    G = np.zeros((r.size, r.size), np.float32)
    for j in range(r.size):
        G[:, j] = filament_flux(r, z, float(r[j]), float(z[j]))
    # a filament's flux at its own position is singular: use the value half a grid cell away instead
    half = 0.5 * float(np.min(np.diff(grid_R)))
    G[np.arange(r.size), np.arange(r.size)] = filament_flux(r + half, z, r, z)
    return G


def cached_grid_greens(cache_dir: Path, grid_R: np.ndarray, grid_Z: np.ndarray) -> np.ndarray:
    """grid_greens, saved beside the cache (greens.npy, 71 MB) and rebuilt only when the grid changes."""
    path, grid_path = cache_dir / "greens.npy", cache_dir / "greens_grid.npz"
    if path.exists() and grid_path.exists():
        g = np.load(grid_path)
        if np.array_equal(g["grid_R"], grid_R) and np.array_equal(g["grid_Z"], grid_Z):
            return np.load(path)
    G = grid_greens(grid_R, grid_Z)
    np.save(path, G)
    np.savez(grid_path, grid_R=grid_R, grid_Z=grid_Z)
    return G
