"""Geometry-only LCFS extraction regions: where a machine's plasma can sit, from its walls and coils alone.

Without the scorer's envelope (rule 3 on MAST) the extractor can close a boundary around a coil or the field null. A
region built from machine geometry bounds the bisection the same way and reads no equilibrium:

- MAST `chamber`: the agnostic geometry box (R 0.2-1.9 m, clear of the centre-column solenoid whose conductors end at
  0.157 m; |Z| <= 1.8 m) minus the shipped PF coil rectangles, dilated by one cell, keeping only cells in line of sight
  of the chamber centre. MAST's P2-P6 coils sit inside the vacuum vessel, so a boundary may not cross or wrap behind
  them. Neither the shipped inputs nor FreeGS `MAST()` carry a MAST limiter outline.
- DIII-D `wall`: the first wall of FreeGS `DIIID()` (freegs-plasma/freegs `freegs/machine.py`, master 5b41fe5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

MastRegion = Literal["none", "box", "chamber"]
MAST_BOX = (0.2, 1.9, 1.8)  # R_min, R_max, |Z|_max [m], as agnostic.MachineFrame.for_machine("MAST")
MAST_CENTRE = (1.05, 0.0)  # the box's centre in R, on the midplane
# FreeGS DIIID() Wall(R, Z) [m], 117 points
DIII_D_WALL_R = """
1.016 1.016 1.016 1.016 1.016 1.016 1.016 1.016 1.016 1.016 1.016 1.012 1.001 1.029 1.042 1.046 1.056 1.097
1.108 1.116 1.134 1.148 1.162 1.181 1.182 1.185 1.19 1.195 1.201 1.209 1.215 1.222 1.228 1.234 1.239 1.242
1.248 1.258 1.263 1.28 1.28 1.28 1.31 1.328 1.361 1.38 1.419 1.419 1.372 1.37167 1.37003 1.36688 1.36719
1.37178 1.37224 1.38662 1.38708 1.40382 1.41127 1.41857 1.421 1.48663 1.4973 1.49762 1.49745 1.49275 1.4926
1.49261 1.49279 1.4934 1.4947 1.49622 1.47981 1.48082 1.48149 1.48646 1.49095 1.50305 1.59697 1.6255 1.63752
1.647 1.785 2.07 2.128 2.245 2.33956 2.34708 2.34913 2.35103 2.35158 2.35125 2.35051 2.34965 2.3487 2.3476
2.3402 2.32942 2.134 1.786 1.768 1.768 1.682 1.372 1.372 1.42 1.42 1.273 1.153 1.016 1.016 1.016 1.016 1.016
1.016 1.016 1.016
"""
DIII_D_WALL_Z = """
0 0.964 0.968 1.001 1.019 1.077 1.07 1.096 1.113 1.138 1.147 1.165 1.217 1.217 1.1624 1.16238 1.1626 1.1645
1.16594 1.16591 1.16896 1.17175 1.17556 1.183 1.1835 1.185 1.188 1.191 1.196 1.202 1.208 1.214 1.221 1.231
1.238 1.244 1.254 1.278 1.29 1.331 1.347 1.348 1.348 1.348 1.348 1.348 1.348 1.31 1.31 1.29238 1.28268
1.25644 1.22955 1.19576 1.19402 1.16487 1.16421 1.15696 1.1573 1.16132 1.164 1.2405 1.23458 1.23428 1.23174
1.2133 1.21061 1.20486 1.20214 1.19642 1.18511 1.1607 1.12426 1.12256 1.12138 1.11692 1.11439 1.11244 1.09489
1.0853 1.07988 1.077 1.077 1.04 0.993 0.709 0.46143 0.41583 0.27218 0.17018 0.07012 -0.03179 -0.14435
-0.21483 -0.32669 -0.38677 -0.45304 -0.47757 -0.973 -1.174 -1.211 -1.25 -1.25 -1.25 -1.329 -1.329 -1.363
-1.363 -1.363 -1.223 -1.223 -0.83 -0.8 -0.415 -0.4 -0.001 0
"""
DIII_D_WALL = np.array([np.array(text.split(), float) for text in (DIII_D_WALL_R, DIII_D_WALL_Z)])


def coil_cells(grid_R: np.ndarray, grid_Z: np.ndarray, coils: dict[str, np.ndarray], dilate: int = 1) -> np.ndarray:
    """(nz, nr) cells covered by any coil rectangle (coil_R, coil_Z, coil_width, coil_height) widened by `dilate`
    grid cells each way."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    dr, dz = dilate * abs(grid_R[1] - grid_R[0]), dilate * abs(grid_Z[1] - grid_Z[0])
    r, z = np.asarray(coils["coil_R"], float), np.asarray(coils["coil_Z"], float)
    hw = np.asarray(coils["coil_width"], float) / 2 + dr
    hh = np.asarray(coils["coil_height"], float) / 2 + dz
    hit = (np.abs(RR[..., None] - r) <= hw) & (np.abs(ZZ[..., None] - z) <= hh)
    return hit.any(-1)


def line_of_sight(
    blocked: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray, centre: tuple[float, float], samples: int = 256
) -> np.ndarray:
    """(nz, nr) cells whose straight segment from `centre` crosses no blocked cell (nearest-cell sampling)."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    t = np.linspace(0.0, 1.0, samples)[:, None, None]
    ir = np.rint((centre[0] + t * (RR - centre[0]) - grid_R[0]) / (grid_R[1] - grid_R[0])).astype(int)
    iz = np.rint((centre[1] + t * (ZZ - centre[1]) - grid_Z[0]) / (grid_Z[1] - grid_Z[0])).astype(int)
    ir, iz = ir.clip(0, len(grid_R) - 1), iz.clip(0, len(grid_Z) - 1)
    return ~blocked[iz, ir].any(0)


def inside_polygon(grid_R: np.ndarray, grid_Z: np.ndarray, poly_R: np.ndarray, poly_Z: np.ndarray) -> np.ndarray:
    """(nz, nr) cell centres inside the closed polygon (even-odd ray casting)."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    r0, z0 = np.asarray(poly_R, float), np.asarray(poly_Z, float)
    r1, z1 = np.roll(r0, -1), np.roll(z0, -1)
    straddle = (z0 > ZZ[..., None]) != (z1 > ZZ[..., None])
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cross = r0 + (ZZ[..., None] - z0) * (r1 - r0) / (z1 - z0)
    return (np.count_nonzero(straddle & (RR[..., None] < r_cross), -1) % 2).astype(bool)


@dataclass(frozen=True)
class Region:
    """A geometry-only extraction region; `mask` None extracts on the whole grid."""

    name: str
    mask: np.ndarray | None

    @classmethod
    def mast(
        cls, name: MastRegion, grid_R: np.ndarray, grid_Z: np.ndarray, coils: dict[str, np.ndarray] | None
    ) -> Region:
        """`none`, `box` (MAST_BOX) or `chamber` (the box minus coils, in line of sight of MAST_CENTRE)."""
        if name == "none":
            return cls(name, None)
        RR, ZZ = np.meshgrid(grid_R, grid_Z)
        r_min, r_max, z_max = MAST_BOX
        box = (RR >= r_min) & (RR <= r_max) & (np.abs(ZZ) <= z_max)
        if name == "box":
            return cls(name, box)
        if name != "chamber" or coils is None:
            raise ValueError(f"unknown MAST region {name!r} (chamber needs the shipped coils)")
        blocked = coil_cells(grid_R, grid_Z, coils)
        return cls(name, box & ~blocked & line_of_sight(blocked, grid_R, grid_Z, MAST_CENTRE))

    @classmethod
    def diii_d_wall(cls, grid_R: np.ndarray, grid_Z: np.ndarray) -> Region:
        return cls("wall", inside_polygon(grid_R, grid_Z, *DIII_D_WALL))
