"""Free-boundary Grad-Shafranov by Picard iteration, batched on the GPU, for machines the network cannot ingest.

The superposition in physics.py puts a fixed ellipse of current in the coil field, so the magnetic axis and the plasma
shape ignore force balance. Here the current follows the flux instead:

    psi = psi_vac + G J                        (G: grid Green's matrix, J: toroidal current per grid cell)
    axis  = max psi inside the scorer's envelope
    bound = max(psi where the plasma would touch the envelope edge, psi at the strongest X-point)
    J    ~ [beta0 R/R0 + (1 - beta0) R0/R] (1 - psiN^alpha)^gamma  on the axis-connected region psiN < 1, sum = Ip

Everything is in "phi" orientation (phi = sign(Ip) psi) so the plasma is a flux maximum on every frame; callers get psi
back in the raw coil orientation. Inputs only: coil currents (via psi_vac), Ip, and an initial current guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

MIX = 0.3  # Picard damping: the share of the new current blended in each iteration


@dataclass(frozen=True)
class Profile:
    alpha: float = 2.0
    gamma: float = 1.0
    beta0: float = 0.5


class GradShafranov:
    """Holds the grid, the envelope and the Green's matrix on the device; `solve` iterates a batch of frames."""

    def __init__(
        self, grid_R: np.ndarray, grid_Z: np.ndarray, envelope: np.ndarray, G: np.ndarray, device: str
    ) -> None:
        self.grid_R, self.grid_Z = grid_R, grid_Z
        self.nz, self.nr = len(grid_Z), len(grid_R)
        self.device = device
        RR, ZZ = np.meshgrid(grid_R, grid_Z)
        self.R = torch.as_tensor(RR.ravel(), dtype=torch.float32, device=device)
        self.Z = torch.as_tensor(ZZ.ravel(), dtype=torch.float32, device=device)
        self.step = torch.tensor([grid_R[1] - grid_R[0], grid_Z[1] - grid_Z[0]], dtype=torch.float32, device=device)
        self.cells = torch.tensor([self.nr - 1, self.nz - 1], dtype=torch.float32, device=device)
        self.inside = torch.as_tensor(envelope.ravel(), dtype=torch.bool, device=device)
        ring = envelope & ~binary_erosion(envelope)
        self.ring = torch.as_tensor(ring.ravel(), dtype=torch.bool, device=device)
        self.Gt = torch.as_tensor(G.T, dtype=torch.float32, device=device)

    def solve(
        self,
        vac: np.ndarray,
        ip_a: np.ndarray,
        j0: np.ndarray,
        profile: Profile,
        iters: int = 20,
        pin: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(psi, J) for B frames: vac (B, N) raw coil flux, ip_a (B,) signed plasma current [A], j0 (B, N) weights.
        With `pin`, each frame's current centroid is held at j0's: with measured (not feedback-controlled) coil
        currents and no vessel currents, a free plasma drifts vertically, so only its shape and profile are solved."""
        sign = torch.as_tensor(np.sign(ip_a) + (ip_a == 0), dtype=torch.float32, device=self.device)[:, None]
        ip = torch.as_tensor(np.abs(ip_a), dtype=torch.float32, device=self.device)[:, None]
        vac_phi = torch.as_tensor(vac, dtype=torch.float32, device=self.device) * sign
        J = torch.as_tensor(j0, dtype=torch.float32, device=self.device) * ip
        target = self.centroid(J)
        for _ in range(iters):
            phi = vac_phi + J @ self.Gt
            j_new = self.current(phi, ip, profile)
            if pin:
                j_new = self.recentre(j_new, target, ip)
            # a frame with no closed surface this round keeps its current rather than collapsing to zero
            J = torch.where(j_new.sum(1, keepdim=True) > 0, (1 - MIX) * J + MIX * j_new, J)
        phi = vac_phi + J @ self.Gt
        return (phi * sign).cpu().numpy(), (J * sign).cpu().numpy()

    def current(self, phi: torch.Tensor, ip: torch.Tensor, profile: Profile) -> torch.Tensor:
        """Profile current on the axis-connected plasma region, normalised to Ip (all zero if there is no region)."""
        neg = torch.finfo(phi.dtype).min
        inside = torch.where(self.inside, phi, torch.full_like(phi, neg))
        phi_ax, ax = inside.max(1, keepdim=True)
        phi_b = torch.maximum(self.limiter_flux(phi), self.xpoint_flux(phi, phi_ax))
        psin = ((phi_ax - phi) / (phi_ax - phi_b).clamp_min(1e-9)).clamp(0, 1)
        region = self.connected(phi > phi_b, ax)
        R0 = (self.R[None] * region).sum(1, keepdim=True) / region.sum(1, keepdim=True).clamp_min(1)
        shape = profile.beta0 * self.R / R0 + (1 - profile.beta0) * R0 / self.R
        j = shape * (1 - psin**profile.alpha).clamp_min(0) ** profile.gamma * region
        return ip * j / j.sum(1, keepdim=True).clamp_min(1e-12)

    def centroid(self, j: torch.Tensor) -> torch.Tensor:
        """(B, 2) current-weighted (R, Z)."""
        w = j / j.sum(1, keepdim=True).clamp_min(1e-12)
        return torch.stack([(w * self.R).sum(1), (w * self.Z).sum(1)], 1)

    def recentre(self, j: torch.Tensor, target: torch.Tensor, ip: torch.Tensor) -> torch.Tensor:
        """Translate each frame's current so its centroid sits at `target` (bilinear resampling), keeping sum = Ip."""
        shift = (self.centroid(j) - target) / self.step * 2 / self.cells
        theta = torch.zeros(len(j), 2, 3, device=j.device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, :, 2] = shift
        grid = F.affine_grid(theta, [len(j), 1, self.nz, self.nr], align_corners=True)
        moved = F.grid_sample(j.view(-1, 1, self.nz, self.nr), grid, align_corners=True).view(len(j), -1)
        return ip * moved / moved.sum(1, keepdim=True).clamp_min(1e-12)

    def limiter_flux(self, phi: torch.Tensor) -> torch.Tensor:
        """Highest flux on the envelope's edge: the surface where the plasma would touch the wall."""
        return torch.where(self.ring, phi, torch.full_like(phi, torch.finfo(phi.dtype).min)).max(1, keepdim=True)[0]

    def xpoint_flux(self, phi: torch.Tensor, phi_ax: torch.Tensor) -> torch.Tensor:
        """Flux at the strongest saddle inside the envelope: a local |grad| minimum with negative Hessian det."""
        p = phi.view(-1, 1, self.nz, self.nr)
        dR = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / 2
        dZ = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / 2
        dRR = p[..., 1:-1, 2:] - 2 * p[..., 1:-1, 1:-1] + p[..., 1:-1, :-2]
        dZZ = p[..., 2:, 1:-1] - 2 * p[..., 1:-1, 1:-1] + p[..., :-2, 1:-1]
        dRZ = (p[..., 2:, 2:] - p[..., 2:, :-2] - p[..., :-2, 2:] + p[..., :-2, :-2]) / 4
        grad2 = dR**2 + dZ**2
        local_min = grad2 <= -F.max_pool2d(-grad2, 3, stride=1, padding=1)
        saddle = local_min & (dRR * dZZ - dRZ**2 < 0)
        saddle = F.pad(saddle, (1, 1, 1, 1)).view(phi.shape[0], -1) & self.inside
        cand = torch.where(saddle & (phi < phi_ax), phi, torch.full_like(phi, torch.finfo(phi.dtype).min))
        return cand.max(1, keepdim=True)[0]

    def connected(self, allowed: torch.Tensor, seed_idx: torch.Tensor) -> torch.Tensor:
        """Float mask of the `allowed` cells 8-connected to each frame's seed cell (flood fill by repeated dilation)."""
        allowed = (allowed & self.inside).view(-1, 1, self.nz, self.nr).float()
        region = torch.zeros_like(allowed).view(allowed.shape[0], -1)
        region.scatter_(1, seed_idx, 1.0)
        region = region.view_as(allowed) * allowed
        for k in range(self.nz + self.nr):
            grown = F.max_pool2d(region, 3, stride=1, padding=1) * allowed
            if k % 8 == 7 and torch.equal(grown, region):  # a host sync, so only every 8 rounds
                break
            region = grown
        return region.view(allowed.shape[0], -1)
