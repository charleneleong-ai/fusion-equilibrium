"""Grid shape model (docs/specs/2026-10-08-grid-shape-model-design.md): a FiLM-conditioned network - a U-Net, or a
Fourier neural operator (docs/specs/2026-10-11-fno-grid-member-design.md) - from the vacuum flux map and the frame's
features to the flux shape - psi centred and scaled inside the scorer's envelope. It predicts shape only
(`predict_shape`); a blend moves it onto a full prediction's level and spread with match_envelope."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from fusioneq.cache import Frames
from fusioneq.model import NetConfig, context_rows
from fusioneq.targets import CHUNK, envelope, unit_shape
from fusioneq.tracking import Run

GRID, PAD = 65, 72  # padded to a multiple of 8 for three downsamplings
EVAL_BATCH = 512  # validation and prediction batch; small enough for a shared GPU
PROGRESS_EVERY_S = 300


@dataclass
class GridConfig:
    arch: Literal["unet", "fno"] = "unet"
    channels: int = 32  # the U-Net's first level, or the FNO's width
    modes: int = 16  # FNO: Fourier modes kept per direction (at most PAD // 2)
    layers: int = 4  # FNO: Fourier layers
    context: int = 4
    film_hidden: int = 512
    env_weight: float = 4.0  # extra loss weight inside the envelope
    gs_weight: float = 0.0  # loss on the Grad-Shafranov operator's error (the current density), in grid units
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10
    batch: int = 256
    stride: int = 2  # every stride-th training frame
    amp: bool = True
    seed: int = 0


def shape_maps(psi: np.ndarray, rows: np.ndarray, dtype: type = np.float16) -> np.ndarray:
    """(N, 65, 65) flux shape (envelope units) of frames `rows` of `psi`, gathered chunk by chunk."""
    out = np.empty((len(rows), GRID, GRID), dtype)
    for i in range(0, len(rows), CHUNK):
        r = rows[i : i + CHUNK]
        flat = np.asarray(psi[r], np.float32).reshape(len(r), -1)
        out[i : i + len(r)] = unit_shape(flat).reshape(-1, GRID, GRID)
    return out


def gs_operator(psi: torch.Tensor, R: torch.Tensor, dR: float, dZ: float) -> torch.Tensor:
    """Delta* psi = d2psi/dR2 - dpsi/dR / R + d2psi/dZ2 (= -mu0 R J_phi) on the interior of (B, Z, R) maps, by central
    differences, scaled by dR^2 into grid units."""
    c = psi[:, 1:-1, 1:-1]
    d2r = psi[:, 1:-1, 2:] - 2 * c + psi[:, 1:-1, :-2]
    d1r = (psi[:, 1:-1, 2:] - psi[:, 1:-1, :-2]) / 2
    d2z = (psi[:, 2:, 1:-1] - 2 * c + psi[:, :-2, 1:-1]) * (dR / dZ) ** 2
    return d2r - d1r * dR / R[1:-1] + d2z


def eval_slices(n: int) -> Iterator[slice]:
    return (slice(i, i + EVAL_BATCH) for i in range(0, n, EVAL_BATCH))


class Block(nn.Module):
    """Two 3x3 convolutions with group norm; the features scale and shift the channels in between (FiLM)."""

    def __init__(self, c_in: int, c_out: int, film_hidden: int) -> None:
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(c_in, c_out, 3, padding=1), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(min(8, c_out), c_out), nn.GroupNorm(min(8, c_out), c_out)
        self.film = nn.Linear(film_hidden, 2 * c_out)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = F.gelu(self.norm1(self.conv1(h)) * (1 + gamma) + beta)
        return F.gelu(self.norm2(self.conv2(h)))


class ConditionedGridNet(nn.Module):
    """Shared front end: an MLP from the features to the FiLM conditioning, and the padded (vacuum shape, R, Z) maps."""

    def __init__(self, n_in: int, film_hidden: int) -> None:
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(n_in, film_hidden), nn.GELU(), nn.Linear(film_hidden, film_hidden), nn.GELU()
        )
        RR, ZZ = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID))
        self.register_buffer("coords", torch.from_numpy(np.stack([RR, ZZ]).astype(np.float32)), persistent=False)

    def padded_maps(self, vac: torch.Tensor) -> torch.Tensor:
        """(B, 3, PAD, PAD) vacuum shape and coordinate maps, zero-padded past the grid."""
        maps = torch.cat([vac[:, None], self.coords.expand(len(vac), -1, -1, -1)], dim=1)
        return F.pad(maps, (0, PAD - GRID, 0, PAD - GRID))


class GridNet(ConditionedGridNet):
    """U-Net over the vacuum-shape map plus R / Z coordinate maps, conditioned on the features at every level."""

    def __init__(self, n_in: int, channels: int, film_hidden: int) -> None:
        super().__init__(n_in, film_hidden)
        c = [channels, 2 * channels, 4 * channels, 8 * channels]
        self.down = nn.ModuleList(
            [Block(3, c[0], film_hidden)] + [Block(c[i], c[i + 1], film_hidden) for i in range(3)]
        )
        self.up = nn.ModuleList([Block(c[i + 1] + c[i], c[i], film_hidden) for i in reversed(range(3))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, vac: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(B, 65, 65) shape from (B, 65, 65) vacuum shape and (B, n_in) features."""
        cond, h = self.cond(x), self.padded_maps(vac)
        skips = []
        for i, block in enumerate(self.down):
            h = block(F.max_pool2d(h, 2) if i else h, cond)
            skips.append(h)
        for block, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            h = block(torch.cat([F.interpolate(h, scale_factor=2), skip], dim=1), cond)
        return self.head(h)[:, 0, :GRID, :GRID]


class SpectralConv(nn.Module):
    """Fourier layer of Li et al. 2020 (arXiv:2010.08895): keep the lowest `modes` frequencies in each direction, mix
    channels there with learned complex weights, transform back at the input's resolution. Runs in float32 (cuFFT
    has no bf16, and half precision only for power-of-two sizes)."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        # [positive, negative] Z frequencies x (in, out) channels x (Z, R) modes
        self.weight = nn.Parameter(torch.randn(2, channels, channels, modes, modes, dtype=torch.cfloat) / channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        m, (n_z, n_r) = self.modes, h.shape[-2:]
        with torch.autocast(h.device.type, enabled=False):
            f = torch.fft.rfft2(h.float())
            out = torch.zeros(len(h), self.weight.shape[2], n_z, n_r // 2 + 1, dtype=f.dtype, device=h.device)
            out[..., :m, :m] = torch.einsum("bizr,iozr->bozr", f[..., :m, :m], self.weight[0])
            out[..., -m:, :m] = torch.einsum("bizr,iozr->bozr", f[..., -m:, :m], self.weight[1])
            return torch.fft.irfft2(out, s=(n_z, n_r))


class FourierBlock(nn.Module):
    """Spectral convolution plus a pointwise linear path, group norm, then the features scale and shift (FiLM)."""

    def __init__(self, channels: int, modes: int, film_hidden: int) -> None:
        super().__init__()
        self.spectral, self.pointwise = SpectralConv(channels, modes), nn.Conv2d(channels, channels, 1)
        self.norm, self.film = nn.GroupNorm(min(8, channels), channels), nn.Linear(film_hidden, 2 * channels)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        return F.gelu(self.norm(self.spectral(h) + self.pointwise(h)) * (1 + gamma) + beta)


class FNONet(ConditionedGridNet):
    """Fourier neural operator: lift the maps to `channels`, `layers` FiLM-conditioned Fourier blocks on the padded
    grid (the zero padding stands in for the non-periodic boundary), project back to one channel."""

    def __init__(self, n_in: int, channels: int, film_hidden: int, modes: int, layers: int) -> None:
        super().__init__(n_in, film_hidden)
        if not 0 < modes <= PAD // 2:
            raise ValueError(f"modes must be in 1..{PAD // 2}, got {modes}")
        self.lift = nn.Conv2d(3, channels, 1)
        self.blocks = nn.ModuleList([FourierBlock(channels, modes, film_hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.GELU(), nn.Conv2d(2 * channels, 1, 1))

    def forward(self, vac: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(B, 65, 65) shape from (B, 65, 65) vacuum shape and (B, n_in) features."""
        cond, h = self.cond(x), self.lift(self.padded_maps(vac))
        for block in self.blocks:
            h = block(h, cond)
        return self.head(h)[:, 0, :GRID, :GRID]


def make_net(cfg: GridConfig, n_in: int) -> ConditionedGridNet:
    match cfg.arch:
        case "unet":
            return GridNet(n_in, cfg.channels, cfg.film_hidden)
        case "fno":
            return FNONet(n_in, cfg.channels, cfg.film_hidden, cfg.modes, cfg.layers)
    raise ValueError(f"unknown grid arch {cfg.arch!r}: expected 'unet' or 'fno'")


class GridPredictor:
    """Fit on cached frames, predict the flux shape for any frame mask; save / load as a directory."""

    net: ConditionedGridNet
    x_mean: np.ndarray
    x_std: np.ndarray
    gs_grid: tuple[torch.Tensor, float, float]  # R, dR, dZ for the Grad-Shafranov loss
    val_loss: float

    def __init__(self, cfg: GridConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        w = torch.from_numpy((1 + cfg.env_weight * envelope()).reshape(GRID, GRID).astype(np.float32))
        self.weight = (w / w.mean()).to(self.device)

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        raw -= self.x_mean
        raw /= self.x_std
        return raw.astype(np.float32, copy=False)

    def raw_inputs(self, fr: Frames, rows: np.ndarray) -> np.ndarray:
        return context_rows(fr, rows, NetConfig(context=self.cfg.context))

    def inputs(self, fr: Frames, rows: np.ndarray) -> np.ndarray:
        """(N, n_in) standardised features for `rows`."""
        return self._normalise(self.raw_inputs(fr, rows))

    def _store(self, X: np.ndarray, fr: Frames, rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        """(features, vacuum shape, target shape) for `rows`, the maps as float16: on the GPU when it has room for
        twice the data (it is shared), else in host memory with each batch copied over."""
        tensors = [torch.from_numpy(a) for a in (X, shape_maps(fr.psi_vac, rows), shape_maps(fr.psi, rows))]
        need = sum(t.numel() * t.element_size() for t in tensors)
        if self.device == "cuda" and torch.cuda.mem_get_info()[0] > 2 * need:
            return tuple(t.to(self.device) for t in tensors)
        if self.device == "cuda":
            logger.info(f"grid data ({need / 1e9:.1f} GB) stays in host memory: the GPU is short of room")
        return tuple(tensors)

    def _forward(self, X: torch.Tensor, vac: torch.Tensor, bf16: bool | None = None) -> torch.Tensor:
        """bf16 autocast on this model's device when `bf16`; by default (training) when cfg.amp on the GPU."""
        X, vac = X.to(self.device), vac.to(self.device)
        bf16 = self.cfg.amp and self.device == "cuda" if bf16 is None else bf16
        with torch.autocast(torch.device(self.device).type, torch.bfloat16, enabled=bf16):
            return self.net(vac.float(), X).float()

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(self.device).float()
        loss = (((pred - target) ** 2) * self.weight).mean()
        if self.cfg.gs_weight:
            R, dR, dZ = self.gs_grid
            err = gs_operator(pred, R, dR, dZ) - gs_operator(target, R, dR, dZ)
            loss = loss + self.cfg.gs_weight * ((err**2) * self.weight[1:-1, 1:-1]).mean()
        return loss

    def fit(
        self, fr: Frames, train: np.ndarray, val: np.ndarray, run: Run | None = None, checkpoint: Path | None = None
    ) -> GridPredictor:
        """Train, keeping the epoch with the lowest validation envelope loss; with `checkpoint`, that model is saved
        there whenever it improves, so a killed run still leaves its best epoch on disk."""
        torch.manual_seed(self.cfg.seed)
        R, Z = np.asarray(fr.grid_R, np.float32), np.asarray(fr.grid_Z, np.float32)
        self.gs_grid = (torch.from_numpy(R).to(self.device), float(R[1] - R[0]), float(Z[1] - Z[0]))
        rows, vrows = np.nonzero(train)[0][:: self.cfg.stride], np.nonzero(val)[0]
        raw = self.raw_inputs(fr, rows)
        self.x_mean, self.x_std = raw.mean(0), raw.std(0) + 1e-6
        data = self._store(self._normalise(raw), fr, rows)
        val_data = self._store(self.inputs(fr, vrows), fr, vrows)
        self.net = make_net(self.cfg, data[0].shape[1]).to(self.device)
        self._train_loop(data, val_data, run, checkpoint)
        return self

    def _train_loop(
        self,
        data: tuple[torch.Tensor, ...],
        val_data: tuple[torch.Tensor, ...],
        run: Run | None,
        checkpoint: Path | None,
    ) -> None:
        c, X = self.cfg, data[0]
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=c.lr, total_steps=c.epochs * -(-len(X) // c.batch), pct_start=0.1
        )
        best_loss, best_state = np.inf, None
        for epoch in range(c.epochs):
            loss = self._train_epoch(epoch, data, opt, sched, run)
            v = self._val_loss(*val_data)
            logger.info(f"grid epoch {epoch + 1}/{c.epochs}: train {loss.item():.4f} val {v:.4f}")
            if run:
                run.log({"epoch": epoch + 1, "train_loss": loss.item(), "val_envelope_loss": v})
            if v < best_loss:
                best_loss, best_state = v, {k: t.detach().clone() for k, t in self.net.state_dict().items()}
                self.val_loss = best_loss
                if checkpoint:
                    self.save(checkpoint)
        if best_state is None:
            raise RuntimeError("grid training never reached a finite validation loss")
        self.net.load_state_dict(best_state)

    def _train_epoch(
        self,
        epoch: int,
        data: tuple[torch.Tensor, ...],
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        run: Run | None,
    ) -> torch.Tensor:
        """One pass over the training frames, reporting every PROGRESS_EVERY_S: an epoch on a shared GPU runs 20-30
        minutes, longer than a run may go silent."""
        X, vac, target = data
        self.net.train()
        batches = torch.randperm(len(X), device=X.device).split(self.cfg.batch)
        start = last = time.monotonic()
        for i, b in enumerate(batches, 1):
            loss = self._loss(self._forward(X[b], vac[b]), target[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (now := time.monotonic()) - last >= PROGRESS_EVERY_S:
                last, eta = now, (now - start) * (len(batches) - i) / i
                logger.info(
                    f"grid epoch {epoch + 1}: batch {i}/{len(batches)} loss {loss.item():.4f}"
                    f" (~{eta / 60:.0f} min left)"
                )
                if run:
                    run.log({"epoch_progress": epoch + i / len(batches), "batch_loss": loss.item()})
        return loss

    @torch.no_grad()
    def _val_loss(self, X: torch.Tensor, vac: torch.Tensor, target: torch.Tensor) -> float:
        self.net.eval()
        total = sum(self._loss(self._forward(X[s], vac[s]), target[s]) * len(X[s]) for s in eval_slices(len(X)))
        return float(total) / len(X)

    @torch.no_grad()
    def predict_shape(self, fr: Frames, mask: np.ndarray, bf16: bool = False) -> np.ndarray:
        """(N, 65, 65) flux shape in envelope units for the frames in `mask`, in float32 on any device: the screens
        score CPU float32 shapes, and a GPU build under bf16 autocast loses the FNO's consistency gain
        (docs/fno-transfer-diagnosis.md). `bf16` reproduces such a build on any device."""
        self.net.eval()
        rows = np.nonzero(mask)[0]
        out = np.empty((len(rows), GRID, GRID), np.float32)
        for i in range(0, len(rows), CHUNK):
            r = rows[i : i + CHUNK]
            X = torch.from_numpy(self.inputs(fr, r)).to(self.device)
            vac = torch.from_numpy(shape_maps(fr.psi_vac, r, np.float32)).to(self.device)
            out[i : i + len(r)] = (
                torch.cat([self._forward(X[s], vac[s], bf16) for s in eval_slices(len(r))]).cpu().numpy()
            )
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "grid.pt")
        np.savez(path / "stats.npz", x_mean=self.x_mean, x_std=self.x_std)
        (path / "grid_config.json").write_text(json.dumps({**asdict(self.cfg), "n_in": len(self.x_mean)}))

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> GridPredictor:
        raw = json.loads((path / "grid_config.json").read_text())
        n_in = raw.pop("n_in")
        m = cls(GridConfig(**raw), device)
        st = np.load(path / "stats.npz")
        m.x_mean, m.x_std = st["x_mean"], st["x_std"]
        m.net = make_net(m.cfg, n_in).to(m.device)
        m.net.load_state_dict(torch.load(path / "grid.pt", map_location=m.device))
        return m


class GridEnsemble:
    """Mean flux shape of several grid models (independently trained seeds make different shape errors)."""

    def __init__(self, members: list[GridPredictor]) -> None:
        self.members = members

    @classmethod
    def load(cls, paths: list[Path], device: str = "cuda") -> GridEnsemble:
        return cls([GridPredictor.load(p, device) for p in paths])

    def predict_shape(self, fr: Frames, mask: np.ndarray) -> np.ndarray:
        total = self.members[0].predict_shape(fr, mask)
        for m in self.members[1:]:
            total += m.predict_shape(fr, mask)
        total /= len(self.members)
        return total
