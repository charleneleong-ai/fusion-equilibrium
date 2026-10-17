"""M2 of docs/specs/2026-09-29-machine-agnostic-network-design.md: a U-Net predicts the plasma current density in the
machine-normalised frame, and psi follows from exact physics on each machine's own grid:

    psi = s * (psi_vac + t * psi_ecoil + r * G J) + c        (Calibration fitted on DIII-D; t = 0 off DIII-D)

Only J (and optionally a per-frame psi offset) is learned, trained on DIII-D, and applied unchanged to MAST.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from fusioneq.agnostic import N_CHANNELS, N_MOMENTS, MachineFrame, machine_current
from fusioneq.cache import Frames
from fusioneq.data import STATIONS
from fusioneq.greens import MU0, cached_grid_greens
from fusioneq.physics import ECOIL_COLUMN, Calibration, ecoil_basis
from fusioneq.scalars import TF_COLUMN, TF_TURNS

# M2b loss ("shape"): the per-frame offset of psi (79% of M2's error, unfixable by any current) is down-weighted
# relative to the offset-free shape error inside the envelope, and a head on the U-Net bottleneck learns the offset.
OFFSET_WEIGHT, FULL_WEIGHT = 0.1, 0.1
# M2d loss probes: the shape loss plus the poloidal-field error ("gradient") or the error in the current shares against
# the labels' own current ("current"), each variance-normalised like the shape term
GRADIENT_WEIGHT, CURRENT_WEIGHT = 1.0, 1.0
N_SCALARS = 3  # mu0 Ip / (h B0), peak log1p Te, peak log1p ne (core Thomson stations, both machines' cache layout)
LossMode = Literal["psi", "shape", "gradient", "current"]
# the per-frame psi offset: none (M2), from the pooled U-Net bottleneck (M2b), or (M2c) from the moments of the vacuum
# field and of its change since the previous frame (vessel eddy currents follow that change) plus the scalars
OffsetMode = Literal["none", "bottleneck", "vacuum"]
N_OFFSET_IN = 2 * N_MOMENTS + N_SCALARS
MIN_B0 = 0.05


@dataclass(frozen=True)
class AgnosticConfig:
    base: int = 32
    levels: int = 4
    epochs: int = 4
    frames_per_epoch: int = 300_000
    batch: int = 64
    lr: float = 1e-3
    seed: int = 0
    loss: LossMode = "psi"  # "psi": MSE over the grid and the envelope (M2); "shape": offset-free shape + weak offset
    offset: OffsetMode = "none"
    correction: bool = False  # M3a: a learned flux field added to psi
    correction_weight: float = 0.0  # M3b: penalty on the correction's energy, in the shape loss's units

    def __post_init__(self) -> None:
        if self.correction_weight and not self.correction:
            raise ValueError("correction_weight needs correction=True")

    @classmethod
    def from_saved(cls, d: dict[str, object]) -> AgnosticConfig:
        """Configs saved before `offset` existed carry `offset_head` (M2b: bottleneck head) or nothing (M2)."""
        d = dict(d)
        if "offset_head" in d:
            d["offset"] = "bottleneck" if d.pop("offset_head") else "none"
        return cls(**d)


class UNet(nn.Module):
    def __init__(
        self, in_ch: int, base: int, levels: int, offset: OffsetMode = "none", correction: bool = False
    ) -> None:
        super().__init__()
        widths = [base * 2**k for k in range(levels)]
        self.down = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.down.append(block(prev, w))
            prev = w
        self.up = nn.ModuleList(block(widths[k + 1] + widths[k], widths[k]) for k in reversed(range(levels - 1)))
        self.out = nn.Conv2d(base, 1 + correction, 1)
        match offset:
            case "bottleneck":
                self.offset = nn.Linear(widths[-1], 1)
            case "vacuum":
                self.offset = nn.Sequential(nn.Linear(N_OFFSET_IN, 64), nn.GELU(), nn.Linear(64, 1))
            case _:
                self.offset = None

    def forward(self, x: torch.Tensor, offset_in: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(outputs (B, 1 + correction, S, S): current logits then the flux correction; per-frame offset (B,)): the head
        reads `offset_in` when given, else the pooled bottleneck; zeros without a head."""
        skips = []
        for k, layer in enumerate(self.down):
            x = layer(x if k == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        if self.offset is None:
            offset = x.new_zeros(len(x))
        else:
            offset = self.offset(offset_in if offset_in is not None else x.mean((2, 3)))[:, 0]
        for layer, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear"), skip], 1))
        return self.out(x), offset


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


@dataclass
class MachineData:
    """One machine's cache as the model sees it: the frame, the Green's matrix, the calibrated E-coil turns (zero off
    DIII-D) and the per-frame scalars, which are small enough to live on the device for every frame."""

    machine: str
    frame: MachineFrame
    Gt: torch.Tensor  # (N, N) transposed Green's matrix
    ecoil: torch.Tensor  # (nz, nr) flux per kA of the E-coil turns psi_vac misses
    ecoil_ka: torch.Tensor  # (T,) E-coil current per frame
    ip_ka: torch.Tensor  # (T,) signed plasma current per frame
    scalars: torch.Tensor  # (T, N_SCALARS)
    offset_features: bool = False  # build the vacuum-offset head's inputs (OffsetMode "vacuum")

    def vacuum(self, fr: Frames, rows: np.ndarray) -> torch.Tensor:
        """(B, nz, nr) calibrated vacuum flux: psi_vac plus the E-coil turns it misses."""
        idx = torch.as_tensor(rows, device=self.Gt.device)
        vac = torch.as_tensor(np.asarray(fr.psi_vac[rows]), device=self.Gt.device).float()
        return vac + self.ecoil_ka[idx, None, None] * self.ecoil

    def inputs(
        self, fr: Frames, rows: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(channels (B, N_CHANNELS + N_SCALARS, S, S), calibrated vacuum (B, nz, nr), |Ip| [A] (B,), offset-head
        inputs (B, N_OFFSET_IN) or None) for rows."""
        idx = torch.as_tensor(rows, device=self.Gt.device)
        vac, ip_ka, scalars = self.vacuum(fr, rows), self.ip_ka[idx], self.scalars[idx]
        ch = self.frame.channels(vac, ip_ka)
        S = ch.shape[-1]
        x = torch.cat([ch, scalars[:, :, None, None].expand(-1, -1, S, S)], 1)
        return x, vac, ip_ka.abs() * 1e3, self.offset_inputs(fr, rows, vac, ch[:, 0], ip_ka, scalars)

    def offset_inputs(
        self,
        fr: Frames,
        rows: np.ndarray,
        vac: torch.Tensor,
        vac_ch: torch.Tensor,
        ip_ka: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor | None:
        """Moments of the vacuum channel and of its change since the previous frame of the shot, plus the scalars."""
        if not self.offset_features:
            return None
        prev = np.maximum(rows - 1, 0)
        prev = np.where(fr.shot_index[prev] == fr.shot_index[rows], prev, rows)
        change = self.frame.to_frame(vac - self.vacuum(fr, prev)) / self.frame.flux_unit(ip_ka)[:, None, None]
        return torch.cat([self.frame.vacuum_moments(vac_ch), self.frame.vacuum_moments(change), scalars], 1)


def frame_scalars(fr: Frames, machine: str, frame: MachineFrame) -> np.ndarray:
    """(T, N_SCALARS): mu0 Ip / (h B0) with B0 from the TF coil, and the peak log1p Te and ne of the core Thomson."""
    n = len(STATIONS[machine]["core"])  # thomson features: core Te, core ne, edge Te, edge ne (data.thomson_features)
    thomson = np.asarray(fr.thomson, np.float32)
    b0 = (
        MU0
        * TF_TURNS[machine]
        * np.abs(np.asarray(fr.currents[:, TF_COLUMN], np.float32))
        * 1e3
        / (2 * np.pi * frame.r_c)
    )
    norm_ip = MU0 * np.abs(np.asarray(fr.ip, np.float32)) * 1e3 / (frame.h * np.maximum(b0, MIN_B0))
    return np.stack([norm_ip, thomson[:, :n].max(1), thomson[:, n : 2 * n].max(1)], 1).astype(np.float32)


def psi_loss(mode: LossMode, pred: torch.Tensor, target: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """Per-frame variance-normalised psi error. "psi": grid MSE + envelope MSE (M2). "shape": inside the envelope the
    error splits exactly into its per-frame mean (offset) and the rest (shape); the offset is down-weighted."""
    err = pred - target
    scale = psi_scale(target)
    grid = (err**2 / scale).mean()
    if mode == "psi":
        return grid + envelope_energy(err, env, scale)
    if mode != "shape":
        raise ValueError(f"unknown loss mode {mode!r}")
    offset = (err * env).sum(1, keepdim=True) / env.sum()
    return envelope_energy(err - offset, env, scale) + OFFSET_WEIGHT * (offset**2 / scale).mean() + FULL_WEIGHT * grid


def psi_scale(target: torch.Tensor) -> torch.Tensor:
    """(B, 1) per-frame psi variance, the normaliser of every psi-space loss term."""
    return target.var(1, keepdim=True).clamp_min(1e-12)


def envelope_energy(field: torch.Tensor, env: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean over frames of a (B, N) field's mean square inside the envelope, over `scale`."""
    return ((field**2 * env).sum(1, keepdim=True) / env.sum() / scale).mean()


def gradient_loss(frame: MachineFrame, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error in the poloidal field (dpsi/dR, dpsi/dZ) inside the envelope over the label's field energy: offset-free,
    and largest where the flux surfaces the scorer contours bend."""
    env = frame.tensor("envelope", pred)
    err, ref = (frame.derivatives(v)[:2] for v in (pred - target, target))
    energy = [sum(((g * env) ** 2).sum((1, 2)) for g in v) for v in (err, ref)]
    return (energy[0] / energy[1].clamp_min(1e-12)).mean()


def current_loss(frame: MachineFrame, shares: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error in the predicted current shares against the label's (MachineFrame.label_current), over the label's."""
    ref = frame.label_current(target)
    return (((shares - ref) ** 2).sum(1) / (ref**2).sum(1).clamp_min(1e-12)).mean()


def correction_penalty(correction: torch.Tensor, target: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """The flux correction's energy inside the envelope, priced like an equal psi error in the shape loss."""
    return envelope_energy(correction, env, psi_scale(target))


class AgnosticPredictor:
    def __init__(self, cfg: AgnosticConfig, calib: Calibration, device: str) -> None:
        self.cfg, self.calib, self.device = cfg, calib, device
        torch.manual_seed(cfg.seed)
        self.net = UNet(N_CHANNELS + N_SCALARS, cfg.base, cfg.levels, cfg.offset, cfg.correction).to(device)

    def psi(
        self,
        data: MachineData,
        x: torch.Tensor,
        vac: torch.Tensor,
        ip_a: torch.Tensor,
        offset_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(psi, J, correction), each (B, nz * nr): psi in EFIT units, J [A per cell] and the learned flux correction
        included in psi (zeros without one). bf16 covers the U-Net only; the Green's step stays fp32."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out, offset = self.net(x, offset_in)
        density = F.softplus(out[:, 0].float()) * x[:, 2]  # channel 2 is the envelope in the frame
        J = machine_current(data.frame, density, ip_a)
        psi = self.calib.apply(vac.reshape(len(vac), -1), J @ data.Gt)
        # the offset and the correction are predicted in the vacuum channel's units, mu0 |Ip| R_c
        unit = self.calib.scale * data.frame.flux_unit(ip_a / 1e3)[:, None]
        correction = torch.zeros_like(psi)
        if self.cfg.correction:
            correction = data.frame.from_frame(out[:, 1].float()).flatten(1) * unit
        return psi + offset.float()[:, None] * unit + correction, J, correction

    def loss(
        self,
        frame: MachineFrame,
        psi: torch.Tensor,
        shares: torch.Tensor,
        correction: torch.Tensor,
        target: torch.Tensor,
        env: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.cfg.loss
        base = psi_loss("psi" if mode == "psi" else "shape", psi, target, env)
        if self.cfg.correction_weight:
            base = base + self.cfg.correction_weight * correction_penalty(correction, target, env)
        match mode:
            case "psi" | "shape":
                return base
            case "gradient":
                return base + GRADIENT_WEIGHT * gradient_loss(frame, psi, target)
            case "current":
                return base + CURRENT_WEIGHT * current_loss(frame, shares, target)
            case other:
                raise ValueError(f"unknown loss mode {other!r}")

    def fit(self, fr: Frames, train: np.ndarray, data: MachineData) -> AgnosticPredictor:
        rows_all = np.nonzero(train)[0]
        rng = np.random.default_rng(self.cfg.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        steps_per_epoch = self.cfg.frames_per_epoch // self.cfg.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.cfg.lr, total_steps=self.cfg.epochs * steps_per_epoch, pct_start=0.05
        )
        env = torch.as_tensor(data.frame.envelope.ravel(), dtype=torch.float32, device=self.device)
        for epoch in range(self.cfg.epochs):
            rows = rng.choice(rows_all, self.cfg.frames_per_epoch, replace=False)
            total = torch.zeros((), device=self.device)
            for i in range(0, steps_per_epoch * self.cfg.batch, self.cfg.batch):
                batch = np.sort(rows[i : i + self.cfg.batch])  # sorted within a batch for memmap reads
                x, vac, ip_a, offset_in = data.inputs(fr, batch)
                target = torch.as_tensor(np.asarray(fr.psi[batch]), device=self.device).float().flatten(1)
                psi, J, correction = self.psi(data, x, vac, ip_a, offset_in)
                loss = self.loss(data.frame, psi, J / ip_a[:, None], correction, target, env)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.detach()
            logger.info(f"epoch {epoch + 1}/{self.cfg.epochs} loss {float(total) / steps_per_epoch:.4f}")
        return self

    @torch.no_grad()
    def predict(self, fr: Frames, rows: np.ndarray, data: MachineData, chunk: int = 512) -> np.ndarray:
        self.net.eval()
        out = np.empty((len(rows), len(fr.grid_Z), len(fr.grid_R)), np.float32)
        for i in range(0, len(rows), chunk):
            x, vac, ip_a, offset_in = data.inputs(fr, rows[i : i + chunk])
            out[i : i + chunk] = self.psi(data, x, vac, ip_a, offset_in)[0].reshape(-1, *out.shape[1:]).cpu().numpy()
        self.net.train()
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "net.pt")
        (path / "config.json").write_text(json.dumps({"cfg": asdict(self.cfg), "calib": asdict(self.calib)}, indent=1))

    @classmethod
    def load(cls, path: Path, device: str) -> AgnosticPredictor:
        d = json.loads((path / "config.json").read_text())
        m = cls(AgnosticConfig.from_saved(d["cfg"]), Calibration(**d["calib"]), device)
        m.net.load_state_dict(torch.load(path / "net.pt", map_location=device))
        return m


def machine_data(
    machine: str,
    fr: Frames,
    cache_dir: Path,
    calib: Calibration,
    device: str,
    coil_file: Path | None = None,
    offset_features: bool = False,
) -> MachineData:
    """MachineData for one machine's cache; the E-coil correction applies only on DIII-D."""
    frame = MachineFrame.for_machine(machine)
    Gt = torch.as_tensor(cached_grid_greens(cache_dir, fr.grid_R, fr.grid_Z).T, device=device)
    shape = (len(fr.grid_Z), len(fr.grid_R))
    ecoil = calib.ecoil_turns * ecoil_basis(fr.grid_R, fr.grid_Z, coil_file) if machine == "DIII-D" else np.zeros(shape)
    ecoil_ka = np.asarray(fr.currents[:, ECOIL_COLUMN], np.float32) if machine == "DIII-D" else np.zeros(len(fr.ip))

    def dev(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, np.float32), device=device)

    scalars = dev(frame_scalars(fr, machine, frame))
    return MachineData(machine, frame, Gt, dev(ecoil), dev(ecoil_ka), dev(fr.ip), scalars, offset_features)
