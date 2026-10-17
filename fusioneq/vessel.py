"""Induced currents in axisymmetric passive conductors (vacuum vessel, coil cases, plates) and their poloidal flux.

Each element k is a toroidal ring of rectangular cross-section. Its current obeys the circuit equation
    sum_j L_kj dI_j/dt + R_k I_k = -2 pi d(psi_ext,k)/dt,
where psi_ext,k (Wb/rad) is the flux the coils and the plasma put through ring k. Mutuals come from the same
filament Green's function as the coil vacuum field (`filament_flux`); self-inductances from Grover's
geometric-mean-distance formula for a thin ring of rectangular section.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from fusioneq.greens import MU0, filament_flux


@dataclass(frozen=True)
class PassiveSet:
    """Passive rings: centre (R, Z), radial width dR, height dZ (m), resistivity (Ohm m); arrays of length N."""

    R: np.ndarray
    Z: np.ndarray
    dR: np.ndarray
    dZ: np.ndarray
    resistivity: np.ndarray

    @property
    def resistance(self) -> np.ndarray:
        return self.resistivity * 2 * np.pi * self.R / (self.dR * self.dZ)

    @cached_property
    def inductance(self) -> np.ndarray:
        """(N, N) H: 2 pi x mutual flux off the diagonal, Grover's ring self-inductance on it."""
        L = 2 * np.pi * filament_flux(self.R[:, None], self.Z[:, None], self.R[None, :], self.Z[None, :])
        gmd = 0.2235 * (self.dR + self.dZ)  # geometric mean distance of a rectangle to itself
        L[np.diag_indices_from(L)] = MU0 * self.R * (np.log(8 * self.R / gmd) - 2)
        return L

    def time_constants(self) -> np.ndarray:
        """Eigen-mode L/R times (s), longest first."""
        rates = np.linalg.eigvals(np.linalg.solve(self.inductance, np.diag(self.resistance))).real
        return np.sort(1 / rates)[::-1]

    def flux_at(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """(*R.shape, N) Wb/rad at points (R, Z) per ampere in each ring."""
        return filament_flux(R[..., None], Z[..., None], self.R, self.Z)

    def induced(self, dt: float, psi_ext: np.ndarray) -> np.ndarray:
        """(T, N) ring currents (A) driven by psi_ext (T, N) Wb/rad sampled every dt seconds, from rest (implicit
        Euler)."""
        L = self.inductance
        lu = lu_factor(L / dt + np.diag(self.resistance))
        out = np.zeros_like(psi_ext, dtype=np.float64)
        for n in range(1, len(psi_ext)):
            out[n] = lu_solve(lu, (L @ out[n - 1] - 2 * np.pi * (psi_ext[n] - psi_ext[n - 1])) / dt)
        return out


def cylinder_shell(
    radius: float, z_min: float, z_max: float, thickness: float, resistivity: float, n: int
) -> PassiveSet:
    """A vertical cylindrical wall cut into n rings of equal height."""
    edges = np.linspace(z_min, z_max, n + 1)
    return PassiveSet(
        R=np.full(n, radius + thickness / 2),
        Z=(edges[:-1] + edges[1:]) / 2,
        dR=np.full(n, thickness),
        dZ=np.diff(edges),
        resistivity=np.full(n, resistivity),
    )


def annular_plate(z: float, r_min: float, r_max: float, thickness: float, resistivity: float, n: int) -> PassiveSet:
    """A horizontal annular plate (vessel lid or floor) cut into n rings of equal radial width."""
    edges = np.linspace(r_min, r_max, n + 1)
    return PassiveSet(
        R=(edges[:-1] + edges[1:]) / 2,
        Z=np.full(n, z + thickness / 2),
        dR=np.diff(edges),
        dZ=np.full(n, thickness),
        resistivity=np.full(n, resistivity),
    )


def join(*parts: PassiveSet) -> PassiveSet:
    return PassiveSet(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("R", "Z", "dR", "dZ", "resistivity")))
