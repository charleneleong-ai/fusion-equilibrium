"""FreeGS's public MAST machine description, as an independent check on the shipped MAST coil geometry.

Source: freegs-plasma/freegs `freegs/machine.py` `MAST()` / `MAST_sym()` (master 5b41fe5, 2026-10-05). FreeGS models
each P-coil as one single-turn filament, the solenoid as N equally spaced filaments at one radius, and no passive
conductors. Bases come out in the units of `coil_basis`: Wb/rad per kA(-turn) of the shipped current column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusioneq.greens import filament_flux

FREEGS_MAST_COILS: dict[str, tuple[float, float]] = {
    "magnetics_p2u_current": (0.49, 1.76),
    "magnetics_p2l_current": (0.49, -1.76),
    "magnetics_p3u_current": (1.1, 1.1),
    "magnetics_p3l_current": (1.1, -1.1),
    "magnetics_p4u_current": (1.51, 1.095),
    "magnetics_p4l_current": (1.51, -1.095),
    "magnetics_p5u_current": (1.66, 0.52),
    "magnetics_p5l_current": (1.66, -0.52),
    "magnetics_p6u_current": (1.5, 0.9),
    "magnetics_p6l_current": (1.5, -0.9),
}
SOLENOID = "magnetics_sol_current"


@dataclass(frozen=True)
class SolenoidSpec:
    radius: float
    z_min: float
    z_max: float
    turns: int

    def flux(self, grid_R: np.ndarray, grid_Z: np.ndarray) -> np.ndarray:
        """(65, 65) flux per kA in every turn, as FreeGS `Solenoid.controlPsi` sums it."""
        RR, ZZ = np.meshgrid(grid_R, grid_Z)
        zs = np.linspace(self.z_min, self.z_max, self.turns)
        return sum(filament_flux(RR, ZZ, self.radius, float(z)) for z in zs) * 1e3


FREEGS_MAST_SOLENOID = SolenoidSpec(0.15, -1.4, 1.4, 100)
FREEGS_MAST_SYM_SOLENOID = SolenoidSpec(0.15, -1.45, 1.45, 100)


def freegs_basis(grid_R: np.ndarray, grid_Z: np.ndarray, columns: list[str]) -> np.ndarray:
    """(C, 65, 65) FreeGS `MAST()` flux per unit column current, aligned with `coil_basis(..., columns)`."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    out = np.zeros((len(columns), *RR.shape), np.float64)
    for j, col in enumerate(columns):
        if col == SOLENOID:
            out[j] = FREEGS_MAST_SOLENOID.flux(grid_R, grid_Z)
        elif col in FREEGS_MAST_COILS:
            out[j] = filament_flux(RR, ZZ, *FREEGS_MAST_COILS[col]) * 1e3
    return out


@dataclass(frozen=True)
class BasisDiff:
    """How two unit-current flux maps differ: grid-mean level vs the remaining shape."""

    level_ratio: float  # mean(ours) / mean(ref)
    level_diff: float  # mean(ours - ref), Wb/rad per kA
    shape_rms: float  # std(ours - ref): the part a per-frame offset cannot absorb
    rel_rms: float  # rms(ours - ref) / rms(ours)

    @classmethod
    def of(cls, ours: np.ndarray, ref: np.ndarray) -> BasisDiff:
        d = ours - ref
        return cls(
            level_ratio=float(ours.mean() / ref.mean()),
            level_diff=float(d.mean()),
            shape_rms=float(d.std()),
            rel_rms=float(np.sqrt((d**2).mean() / (ours**2).mean())),
        )
