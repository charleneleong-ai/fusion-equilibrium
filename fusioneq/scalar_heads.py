"""Scalar heads for the finishing step: a small residual MLP from the members' own inputs (coil currents, Ip, Thomson,
temporal context, shot history) to the scorer's seven consistency scalars of the label flux (cons_targets.npz, DIII-D
training frames only). Trained in float32 with TF32 off, so its predictions do not depend on the GPU."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from fusioneq.cache import Frames
from fusioneq.model import NetConfig, ResidualMLP, context_rows

N_SCALARS = 7


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 512
    layers: int = 4
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    clip: float = 1.0  # gradient-norm clip
    batch: int = 2048
    context: int = 4
    thomson_hr: bool = True
    history: bool = True
    residual: bool = False  # also read the map's own scalars (an input) and predict truth - map reading
    seed: int = 0

    @property
    def inputs(self) -> NetConfig:
        return NetConfig(context=self.context, thomson_hr=self.thomson_hr, history=self.history)


def exact_float32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ScalarHeads:
    def __init__(self, cfg: HeadConfig, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _raw(self, fr: Frames, rows: np.ndarray, reading: np.ndarray | None) -> np.ndarray:
        """Network inputs before normalisation; residual heads append the map's reading (NaN -> its fit mean)."""
        raw = context_rows(fr, rows, self.cfg.inputs)
        if not self.cfg.residual:
            return raw
        if reading is None:
            raise ValueError("residual heads need the map's reading of each frame")
        return np.concatenate([raw, np.where(np.isfinite(reading), reading, self.r_mean).astype(np.float32)], 1)

    def _x(self, fr: Frames, rows: np.ndarray, reading: np.ndarray | None) -> torch.Tensor:
        x = (self._raw(fr, rows, reading) - self.x_mean) / self.x_std
        return torch.from_numpy(x.astype(np.float32))

    def fit(self, fr: Frames, rows: np.ndarray, Y: np.ndarray, reading: np.ndarray | None = None) -> ScalarHeads:
        """rows (N,) training frames, Y (N, 7) their scorer scalars (NaN where the derivation failed); residual heads
        also take the map's (N, 7) reading of those frames and learn Y - reading."""
        exact_float32()
        torch.manual_seed(self.cfg.seed)
        if self.cfg.residual:
            self.r_mean = np.nanmean(reading, 0)
            Y = Y - reading
        raw = self._raw(fr, rows, reading)
        self.x_mean, self.x_std = raw.mean(0), raw.std(0) + 1e-6
        self.y_mean, self.y_std = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-12
        X = torch.from_numpy(((raw - self.x_mean) / self.x_std).astype(np.float32)).to(self.device)
        y = (Y - self.y_mean) / self.y_std
        ok = torch.from_numpy(np.isfinite(y).astype(np.float32)).to(self.device)
        yt = torch.from_numpy(np.nan_to_num(y).astype(np.float32)).to(self.device)
        c = self.cfg
        self.net = ResidualMLP(X.shape[1], N_SCALARS, c.hidden, c.layers, c.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        per_epoch = -(-len(X) // c.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.lr, total_steps=c.epochs * per_epoch, pct_start=0.1)
        gen = torch.Generator(device=self.device).manual_seed(c.seed)
        for epoch in range(c.epochs):
            self.net.train()
            tot = 0.0
            for idx in torch.randperm(len(X), device=self.device, generator=gen).split(c.batch):
                loss = (((self.net(X[idx]) - yt[idx]) ** 2) * ok[idx]).sum() / ok[idx].sum().clamp_min(1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.clip)
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(idx)
            logger.info(f"heads epoch {epoch + 1}/{c.epochs} loss {tot / len(X):.4f}")
        return self

    @torch.no_grad()
    def predict(
        self, fr: Frames, rows: np.ndarray, reading: np.ndarray | None = None, chunk: int = 65536
    ) -> np.ndarray:
        """(N, 7) predicted scalars, float64 (residual heads: reading + predicted residual)."""
        exact_float32()
        self.net.eval()
        out = []
        for i in range(0, len(rows), chunk):
            r = None if reading is None else reading[i : i + chunk]
            out.append(self.net(self._x(fr, rows[i : i + chunk], r).to(self.device)).cpu().numpy())
        pred = np.concatenate(out).astype(np.float64) * self.y_std + self.y_mean
        return reading + pred if self.cfg.residual else pred

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "net.pt")
        names = ("x_mean", "x_std", "y_mean", "y_std") + (("r_mean",) if self.cfg.residual else ())
        stats = {k: getattr(self, k) for k in names}
        np.savez(path / "stats.npz", **stats)
        (path / "config.json").write_text(json.dumps(asdict(self.cfg), indent=1))

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> ScalarHeads:
        h = cls(HeadConfig(**json.loads((path / "config.json").read_text())), device)
        for k, v in np.load(path / "stats.npz").items():
            setattr(h, k, v)
        c = h.cfg
        h.net = ResidualMLP(len(h.x_mean), N_SCALARS, c.hidden, c.layers, c.dropout).to(h.device)
        h.net.load_state_dict(torch.load(path / "net.pt", map_location=h.device))
        return h
