"""A machine-normalised frame for a network that transfers from DIII-D to MAST (M0/M1 of
docs/specs/2026-09-29-machine-agnostic-network-design.md).

Each machine's 65 x 65 grid maps to a common 128 x 128 frame in envelope-normalised coordinates x = (R - R_c) / h,
y = Z / h, where R_c and h are the centre and half-width in R of the scorer's envelope. The network sees every machine
in this frame; its current is mapped back to the machine grid for the exact psi = psi_vac + G J step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F

from fusioneq.greens import MU0
from fusioneq.physics import CHUNK
from fusioneq.plasma import PlasmaShape
from fusioneq.score import MASKS

FRAME = 128  # frame-check: 64 loses up to 0.4% of MAST's vacuum flux range round trip, 128 keeps p50 under 0.1%
MAST_GRID = (np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65))  # the shipped MAST EFIT grid
X_HALF, Y_HALF = 1.1, 2.2  # frame extent in units of h: covers both envelopes (|y| <= 2.0 on DIII-D and on MAST)
N_MOMENTS = 6  # vacuum (homogeneous Grad-Shafranov) fields, see vacuum_fields
N_CHANNELS = 4  # MachineFrame.channels: vacuum flux, R / R_c, envelope, y


def vacuum_fields(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(N_MOMENTS, ...) solutions of the homogeneous Grad-Shafranov equation in r = R / R_c, z = Z / R_c."""
    return np.stack([np.ones_like(r), r**2, z, r**2 * z, r**4 - 4 * r**2 * z**2, r**2 * np.log(r) - z**2])


@dataclass(frozen=True)
class MachineFrame:
    grid_R: np.ndarray
    grid_Z: np.ndarray
    envelope: np.ndarray  # (nz, nr) bool, the scorer's mask
    r_c: float
    h: float
    size: int = FRAME

    @classmethod
    def for_machine(cls, machine: str, size: int = FRAME) -> MachineFrame:
        """DIII-D: the scorer's envelope. MAST: a geometry box clear of the centre-column solenoid (R >= 0.2 m; its
        conductors end at 0.16 m) and the grid edges - the scorer's MAST envelope may not shape a MAST
        prediction (rule 3)."""
        if machine == "MAST":
            grid_R, grid_Z = MAST_GRID
            RR, ZZ = np.meshgrid(grid_R, grid_Z)
            envelope = (RR >= 0.2) & (RR <= 1.9) & (np.abs(ZZ) <= 1.8)
        else:
            m = np.load(MASKS[machine])
            grid_R, grid_Z, envelope = m["grid_R"], m["grid_Z"], m["mask_coarse"].astype(bool)
        R = np.meshgrid(grid_R, grid_Z)[0][envelope]
        return cls(grid_R, grid_Z, envelope, float(R.max() + R.min()) / 2, float(R.max() - R.min()) / 2, size)

    @cached_property
    def frame_RZ(self) -> tuple[np.ndarray, np.ndarray]:
        """(size, size) physical R and Z of the frame's cells, rows = y (Z), cols = x (R)."""
        X, Y = np.meshgrid(np.linspace(-X_HALF, X_HALF, self.size), np.linspace(-Y_HALF, Y_HALF, self.size))
        return self.r_c + self.h * X, self.h * Y

    @cached_property
    def grids(self) -> dict[str, np.ndarray]:
        """grid_sample coordinates: 'to' samples the machine grid at the frame's cells, 'from' the reverse."""
        R, Z = self.frame_RZ
        to = np.stack(
            [
                2 * (R - self.grid_R[0]) / (self.grid_R[-1] - self.grid_R[0]) - 1,
                2 * (Z - self.grid_Z[0]) / (self.grid_Z[-1] - self.grid_Z[0]) - 1,
            ],
            -1,
        )
        Rm, Zm = np.meshgrid(self.grid_R, self.grid_Z)
        back = np.stack([(Rm - self.r_c) / self.h / X_HALF, Zm / self.h / Y_HALF], -1)
        return {"to": to, "from": back}

    @cached_property
    def device_cache(self) -> dict[tuple[str, str, torch.dtype], torch.Tensor]:
        return {}

    def tensor(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """A static array (sampling grids, masks, fixed channels) on `like`'s device and dtype, built once."""
        key = (name, str(like.device), like.dtype)
        if key not in self.device_cache:
            self.device_cache[key] = torch.as_tensor(self.static_array(name), dtype=like.dtype, device=like.device)
        return self.device_cache[key]

    def static_array(self, name: str) -> np.ndarray:
        if name in self.grids:
            return self.grids[name]
        if name == "envelope":
            return self.envelope
        if name == "R":
            return self.grid_R
        R, Z = self.frame_RZ
        env = self.to_frame(torch.as_tensor(self.envelope[None], dtype=torch.float64))[0].numpy() > 0.5
        if name == "moment_pinv":
            return self.moment_pinv(R / self.r_c, Z / self.r_c, env)
        return {"frame_envelope": env, "static": np.stack([R / self.r_c, env, Z / self.h])}[name]

    @staticmethod
    def moment_pinv(r: np.ndarray, z: np.ndarray, env: np.ndarray) -> np.ndarray:
        """(N_MOMENTS, S * S): least-squares projection of a frame field onto the vacuum fields inside the envelope."""
        basis = vacuum_fields(r, z).reshape(N_MOMENTS, -1).T * env.reshape(-1, 1)
        return np.linalg.pinv(basis)

    def vacuum_moments(self, field: torch.Tensor) -> torch.Tensor:
        """(B, N_MOMENTS) coefficients of a (B, S, S) frame field on the vacuum fields, fitted inside the envelope."""
        return field.flatten(1) @ self.tensor("moment_pinv", field).T

    def to_frame(self, field: torch.Tensor) -> torch.Tensor:
        """(B, nz, nr) on the machine grid -> (B, size, size) in the frame; cells beyond the machine grid read 0."""
        return self.sample(field, self.tensor("to", field))

    def from_frame(self, field: torch.Tensor) -> torch.Tensor:
        """(B, size, size) in the frame -> (B, nz, nr) on the machine grid; cells beyond the frame read 0."""
        return self.sample(field, self.tensor("from", field))

    @staticmethod
    def sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.expand(len(field), *grid.shape)
        return F.grid_sample(field[:, None], grid, mode="bicubic", align_corners=True, padding_mode="zeros")[:, 0]

    def derivatives(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(dpsi/dR, dpsi/dZ, Delta* psi), each (B, nz, nr), for (B, nz * nr) grid fields by central differences;
        the grid's outer ring reads 0. Delta* psi = R d/dR (1/R dpsi/dR) + d2psi/dZ2 = -mu0 R J."""
        p = psi.reshape(len(psi), len(self.grid_Z), len(self.grid_R))
        dr, dz = float(self.grid_R[1] - self.grid_R[0]), float(self.grid_Z[1] - self.grid_Z[0])
        c = p[:, 1:-1, 1:-1]
        p_r = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2 * dr)
        p_z = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / (2 * dz)
        p_rr = (p[:, 1:-1, 2:] - 2 * c + p[:, 1:-1, :-2]) / dr**2
        p_zz = (p[:, 2:, 1:-1] - 2 * c + p[:, :-2, 1:-1]) / dz**2
        R = self.tensor("R", psi)[1:-1]
        return tuple(F.pad(v, (1, 1, 1, 1)) for v in (p_r, p_z, p_rr - p_r / R + p_zz))

    def label_current(self, psi: torch.Tensor) -> torch.Tensor:
        """(B, nz * nr) shares (summing to 1) of the plasma current per machine-grid cell that produce a label psi,
        from the Grad-Shafranov operator: J ~ -Delta* psi / R inside the envelope, oriented positive, clamped at zero
        (finite-difference noise) like the network's current."""
        j = (-self.derivatives(psi)[2] / self.tensor("R", psi)).flatten(1)
        j = j * self.tensor("envelope", psi).flatten()
        j = (j * j.sum(1, keepdim=True).sign()).clamp_min(0)
        return j / j.sum(1, keepdim=True).clamp_min(1e-12)

    def flux_unit(self, ip_ka: torch.Tensor) -> torch.Tensor:
        """(B,) mu0 |Ip| R_c: the flux scale that makes psi dimensionless on any machine (Ip floored at 1 kA)."""
        return MU0 * ip_ka.abs().clamp_min(1.0) * 1e3 * self.r_c

    def channels(self, psi_vac: torch.Tensor, ip_ka: torch.Tensor) -> torch.Tensor:
        """(B, 4, size, size) dimensionless inputs: vacuum flux / (mu0 |Ip| R_c), R / R_c, envelope, y."""
        vac = self.to_frame(psi_vac) / self.flux_unit(ip_ka)[:, None, None]
        static = self.tensor("static", psi_vac)
        return torch.cat([vac[:, None], static[None].expand(len(vac), -1, -1, -1)], 1)


def ellipse_current(frame: MachineFrame, geom: np.ndarray, shape: PlasmaShape, device: str = "cpu") -> torch.Tensor:
    """(B, size, size) parabolic current density on the frame for rows (R0, a, Z0): plasma.current_density's ellipse,
    evaluated on the frame's cells so it follows the same frame -> machine path as a learned current."""
    R, Z = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in frame.frame_RZ)
    r0, a, z0 = torch.as_tensor(geom[:, :3], dtype=torch.float32, device=device)[:, :, None, None].unbind(1)
    rho2 = ((R - r0) / a) ** 2 + ((Z - z0) / (shape.kappa * a)) ** 2
    return (1 - rho2).clamp_min(0) ** shape.alpha


def machine_current(frame: MachineFrame, density: torch.Tensor, ip_a: torch.Tensor) -> torch.Tensor:
    """(B, nz * nr) current per machine-grid cell [A] from a frame current density: resampled, clamped at zero (bicubic
    overshoot), confined to the envelope and scaled so each frame carries exactly its Ip."""
    j = (frame.from_frame(density).clamp_min(0) * frame.tensor("envelope", density)).reshape(len(density), -1)
    return ip_a[:, None] * j / j.sum(1, keepdim=True).clamp_min(1e-12)


def pipeline_plasma(
    frame: MachineFrame, geom: np.ndarray, ip_ka: np.ndarray, Gt: torch.Tensor, shape: PlasmaShape
) -> np.ndarray:
    """(N, nz * nr) plasma flux G J of the ellipse current, routed through the frame, in chunks on Gt's device."""
    out = np.empty((len(geom), Gt.shape[0]), np.float32)
    for i in range(0, len(geom), CHUNK):
        sl = slice(i, i + CHUNK)
        density = ellipse_current(frame, geom[sl], shape, str(Gt.device))
        ip_a = torch.as_tensor(np.asarray(ip_ka[sl], np.float32) * 1e3, device=Gt.device)
        out[sl] = (machine_current(frame, density, ip_a) @ Gt).cpu().numpy()
    return out
