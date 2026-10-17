"""Neural predictor: features -> PCA coefficients of the plasma residual -> psi, with a scalar head."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from fusioneq.cache import Frames, ramp_frames, shot_bounds
from fusioneq.targets import envelope, fit_pca, r2_psi, reconstruct, transform_all, transform_shape
from fusioneq.tracking import Run


class Predictor(Protocol):
    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class NetConfig:
    n_components: int = 128
    hidden: int = 1024
    layers: int = 4
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch: int = 2048
    use_vacuum: bool = True
    context: int = 0  # extra frames on each side whose features are concatenated (temporal context)
    amp: bool = False  # bf16 autocast for the forward/backward pass
    coef_loss: str = "std"  # std: equal weight per coefficient; pixel: variance-weighted = MSE on psi; mix: half each
    region_weight: float = 0.0  # extra loss on psi inside the scorer's envelope, where the LCFS and scalars are derived
    seed: int = 0
    thomson_hr: bool = False  # full-resolution, time-averaged Thomson inputs (cache thomson_hr.npy)
    history: bool = False  # past-looking shot context per frame (Frames.history)
    temporal_layers: int = 0  # >0: TemporalNet, dilated convolutions over the whole shot after the per-frame MLP
    window: int = 64  # frames per training window for the temporal net
    ramp_weight: float = 1.0  # loss weight of ramp-up / ramp-down frames (cache.ramp_frames); 1 = unweighted
    normalize: bool = False  # PCA of the flux shape (targets: psi centred / scaled in the envelope) + its 2 stats


class ResidualMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
                for _ in range(layers)
            ]
        )
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class TemporalBlock(nn.Module):
    """Depthwise dilated convolution over time (both directions, kernel 5) followed by a per-frame channel mix."""

    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(
            hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden, padding_mode="replicate"
        )
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mix(self.conv(self.norm(h).transpose(1, 2)).transpose(1, 2))


class TemporalNet(nn.Module):
    """The per-frame ResidualMLP trunk, then residual TemporalBlocks with dilations 1, 2, 4, ... over a shot's frames:
    (B, T, n_in) -> (B, T, n_out), each frame seeing +-2 * (2**temporal_layers - 1) neighbours."""

    def __init__(self, n_in: int, n_out: int, hidden: int, layers: int, dropout: float, temporal_layers: int) -> None:
        super().__init__()
        self.frame = ResidualMLP(n_in, hidden, hidden, layers, dropout)
        self.blocks = nn.ModuleList([TemporalBlock(hidden, 2**i, dropout) for i in range(temporal_layers)])
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.frame(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def shot_windows(shots: np.ndarray, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(n, length) row indices of windows tiling every shot from a random offset, and their validity: positions
    outside the window's shot repeat its edge frame and are invalid, so each frame is valid exactly once."""
    starts, ends = shot_bounds(shots)
    offset = int(rng.integers(length))
    w0 = [np.arange(s - offset, e, length) for s, e in zip(starts, ends, strict=True)]
    shot = np.repeat(np.arange(len(starts)), [len(w) for w in w0])
    idx = np.concatenate(w0)[:, None] + np.arange(length)
    lo, hi = starts[shot][:, None], ends[shot][:, None]
    return np.clip(idx, lo, hi - 1), (idx >= lo) & (idx < hi)


def max_windows(shots: np.ndarray, length: int) -> int:
    """Most windows shot_windows can return for these shots, over its random offsets."""
    starts, ends = shot_bounds(shots)
    lengths = ends - starts
    return max(int(np.sum(-(-(lengths + o) // length))) for o in range(length))


def masked_mean(x: torch.Tensor, v: torch.Tensor | None) -> torch.Tensor:
    """Mean of `x` over all elements, or (v given) over frames weighted by `v` after averaging x's last axis."""
    return x.mean() if v is None else (x.mean(-1) * v).sum() / v.sum()


def context_features(X: np.ndarray, shot_index: np.ndarray, k: int) -> np.ndarray:
    """Concatenate features of the k previous and k following frames within the same shot (edge-padded)."""
    if k == 0:
        return X
    parts = [X]
    for d in list(range(-k, 0)) + list(range(1, k + 1)):
        idx = np.clip(np.arange(len(X)) + d, 0, len(X) - 1)
        same = shot_index[idx] == shot_index
        idx = np.where(same, idx, np.arange(len(X)))
        parts.append(X[idx])
    return np.concatenate(parts, axis=1)


def context_rows(fr: Frames, rows: np.ndarray, cfg: NetConfig) -> np.ndarray:
    """The network's raw inputs for `rows`: context_features(fr.features(), fr.shot_index, cfg.context)[rows], built
    from only those rows and their neighbours, then Frames.history once per frame when cfg.history."""
    k, thomson_hr = cfg.context, cfg.thomson_hr
    parts = [fr.features(rows, thomson_hr)]
    for d in [*range(-k, 0), *range(1, k + 1)]:
        idx = np.clip(rows + d, 0, len(fr.shot_index) - 1)
        parts.append(fr.features(np.where(fr.shot_index[idx] == fr.shot_index[rows], idx, rows), thomson_hr))
    if cfg.history:
        parts.append(fr.history[rows])
    return np.concatenate(parts, axis=1)


class NeuralPredictor:
    """Fit on cached frames; predict (psi, [q95, betaN]) for any frame mask; save/load as a directory."""

    def __init__(self, cfg: NetConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        if cfg.normalize and cfg.region_weight > 0:
            raise ValueError("the region loss is defined on residual PCA coefficients, not on normalized ones")

    @property
    def n_coef(self) -> int:
        """Outputs before the q95 / betaN head: the PCA coefficients, plus (mean, log spread) when normalized."""
        return self.cfg.n_components + 2 * self.cfg.normalize

    def coef_block(self, fr: Frames, rows: np.ndarray) -> np.ndarray:
        """(N, n_coef) the targets in front of the scalar head, unscaled."""
        if self.cfg.normalize:
            return np.concatenate(transform_shape(self.pca, fr, rows), 1)
        return transform_all(self.pca, fr, rows, self.cfg.use_vacuum)

    def inputs(self, fr: Frames, rows: np.ndarray) -> np.ndarray:
        """(N, n_in) normalized network inputs for `rows`."""
        return ((context_rows(fr, rows, self.cfg) - self.x_mean) / self.x_std).astype(np.float32)

    def fit(
        self, fr: Frames, train: np.ndarray, val: np.ndarray | None = None, run: Run | None = None
    ) -> NeuralPredictor:
        torch.manual_seed(self.cfg.seed)
        rows = np.nonzero(train)[0]
        raw = context_rows(fr, rows, self.cfg)
        self.x_mean, self.x_std = raw.mean(0), raw.std(0) + 1e-6
        self.pca = fit_pca(fr, rows, self.cfg.n_components, self.cfg.use_vacuum, normalize=self.cfg.normalize)
        coef = self.coef_block(fr, rows)
        self.c_std = coef.std(0) + 1e-6
        self.c_mean = np.r_[np.zeros(self.cfg.n_components), coef[:, self.cfg.n_components :].mean(0)]
        scal = fr.scalars[rows][:, :2].astype(np.float32)
        self.s_mean, self.s_std = np.nanmean(scal, 0), np.nanstd(scal, 0) + 1e-6
        X, y, s_ok = self._training_arrays(fr, rows, coef, raw)
        self.net = self._make_net(X.shape[1])
        weights = self._frame_weights(fr, rows)
        self._train_loop(X, y, s_ok, fr, val, run, shots=fr.shot_index[rows], weights=weights)
        return self

    def _frame_weights(self, fr: Frames, rows: np.ndarray) -> np.ndarray | None:
        """Per-frame loss weights for `rows`: cfg.ramp_weight on ramp-up / ramp-down frames; None when unweighted."""
        if self.cfg.ramp_weight == 1.0:
            return None
        ramp = ramp_frames(fr.shot_index[rows], fr.efit_times[rows], fr.ip[rows])
        return np.where(ramp, self.cfg.ramp_weight, 1.0).astype(np.float32)

    def _make_net(self, n_in: int) -> nn.Module:
        c, n_out = self.cfg, self.n_coef + 2
        if c.temporal_layers:
            return TemporalNet(n_in, n_out, c.hidden, c.layers, c.dropout, c.temporal_layers).to(self.device)
        return ResidualMLP(n_in, n_out, c.hidden, c.layers, c.dropout).to(self.device)

    @property
    def windows_per_step(self) -> int:
        return max(1, self.cfg.batch // self.cfg.window)

    def _batches(
        self, n: int, shots: np.ndarray | None, rng: np.random.Generator, weights: torch.Tensor | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        """One epoch of (row indices, per-frame weight): shuffled frames for the MLP (weight None unless `weights`);
        for the temporal net, (B, window) shot windows with about cfg.batch frames per step, weighted by validity."""
        if not self.cfg.temporal_layers:
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.cfg.batch):
                idx = perm[i : i + self.cfg.batch]
                yield idx, None if weights is None else weights[idx]
            return
        idx, valid = shot_windows(shots, self.cfg.window, rng)
        order = rng.permutation(len(idx))
        idx_t = torch.from_numpy(idx[order]).to(self.device)
        valid_t = torch.from_numpy(valid[order].astype(np.float32)).to(self.device)
        if weights is not None:
            valid_t = valid_t * weights[idx_t]
        per = self.windows_per_step
        for i in range(0, len(order), per):
            yield idx_t[i : i + per], valid_t[i : i + per]

    def _training_arrays(
        self, fr: Frames, rows: np.ndarray, coef: np.ndarray, raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(normalised inputs, scaled targets, scalar-label mask) for `rows` from their raw context features `raw`,
        with this model's statistics."""
        X = ((raw - self.x_mean) / self.x_std).astype(np.float32)
        scal = fr.scalars[rows][:, :2].astype(np.float32)
        y = np.concatenate([(coef - self.c_mean) / self.c_std, np.nan_to_num((scal - self.s_mean) / self.s_std)], 1)
        return X, y.astype(np.float32), np.isfinite(scal).all(1).astype(np.float32)

    def _train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        s_ok: np.ndarray,
        fr: Frames,
        val: np.ndarray | None,
        run: Run | None = None,
        extra_loss: Callable[[NeuralPredictor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        shots: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """extra_loss(model, X, y) adds a term to every step's loss (fusioneq.consistency.fine_tune); `shots` (the
        rows' shot ids) is needed by the temporal net's windows."""
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        steps = self.cfg.epochs * (len(X) // self.cfg.batch + 1)
        if self.cfg.temporal_layers:
            steps = self.cfg.epochs * -(-max_windows(shots, self.cfg.window) // self.windows_per_step)
        rng = np.random.default_rng(self.cfg.seed)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.lr, total_steps=steps, pct_start=0.1)
        Xt, yt, st = (torch.from_numpy(a).to(self.device) for a in (X, y, s_ok))
        wt = None if weights is None else torch.from_numpy(weights).to(self.device)
        nc = self.n_coef
        w = self.coef_weights(device=Xt.device)
        region = self.region_metric(device=Xt.device) if self.cfg.region_weight > 0 else None
        for epoch in range(self.cfg.epochs):
            self.net.train()
            tot = torch.zeros((), device=self.device)
            mass = torch.zeros((), device=self.device)
            for idx, v in self._batches(len(Xt), shots, rng, wt):
                pred, target = self.forward(Xt[idx]), yt[idx]
                d = pred[..., :nc] - target[..., :nc]
                loss_c = masked_mean((d**2) * w, v)
                if region is not None:
                    loss_c = loss_c + self.cfg.region_weight * masked_mean((d @ region) * d, v)
                sv = st[idx] if v is None else st[idx] * v
                loss_s = (((pred[..., nc:] - target[..., nc:]) ** 2).mean(-1) * sv).sum() / sv.sum().clamp_min(1)
                loss = loss_c + 0.5 * loss_s
                if extra_loss is not None:
                    loss = loss + extra_loss(self, Xt, yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                weight = len(idx) if v is None else v.sum()
                tot += loss.detach() * weight
                mass += weight
            metrics = {"train_loss": float(tot / mass), "lr": sched.get_last_lr()[0]}
            if val is not None and (epoch % 5 == 4 or epoch == self.cfg.epochs - 1):
                metrics["holdout_r2_psi"] = self.r2_psi(fr, val)
            logger.info(f"epoch {epoch + 1}/{self.cfg.epochs} " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            if run is not None:
                run.log(metrics, step=epoch + 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Network output in fp32, bf16 autocast inside when cfg.amp."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.amp and self.device == "cuda"):
            return self.net(X).float()

    def coef_weights(self, device: str | torch.device) -> torch.Tensor:
        """Per-coefficient loss weights, mean 1 over the PCA coefficients. `pixel` weights by variance, so the loss
        equals pixel MSE on psi; the normalized target's two stats weigh 1."""
        pixel = self.c_std[: self.cfg.n_components].astype(np.float32) ** 2
        pixel = pixel / pixel.mean()
        w = {"std": np.ones_like(pixel), "pixel": pixel, "mix": 0.5 + 0.5 * pixel}[self.cfg.coef_loss]
        w = np.r_[w, np.ones(self.n_coef - self.cfg.n_components)]
        return torch.from_numpy(w.astype(np.float32)).to(device)

    def region_metric(self, device: str | torch.device) -> torch.Tensor:
        """(nc, nc) M: d^T M d is the psi MSE inside the envelope for scaled-coefficient error d; trace nc."""
        B = self.pca.components_[:, envelope()] * self.c_std[:, None]
        M = B @ B.T
        return torch.from_numpy((M * len(M) / np.trace(M)).astype(np.float32)).to(device)

    @torch.no_grad()
    def coefficients(self, fr: Frames, mask: np.ndarray, X: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(N, n_coef) unscaled coefficient block (PCA, plus mean / log spread when normalized) and (N, 2) scalars;
        `X` reuses inputs already built by `inputs`."""
        self.net.eval()
        rows = np.nonzero(mask)[0]
        X = torch.from_numpy(self.inputs(fr, rows) if X is None else X).to(self.device)
        if self.cfg.temporal_layers:  # one shot at a time, so each frame sees only its own shot
            bounds = zip(*shot_bounds(fr.shot_index[rows]), strict=True)
            out = torch.cat([self.net(X[s:e][None])[0] for s, e in bounds]).cpu().numpy()
        else:
            out = torch.cat([self.net(X[i : i + 8192]) for i in range(0, len(X), 8192)]).cpu().numpy()
        nc = self.n_coef
        block = (out[:, :nc] * self.c_std + self.c_mean).astype(np.float32)
        return block, (out[:, nc:] * self.s_std + self.s_mean).astype(np.float32)

    def decode(self, block: np.ndarray, fr: Frames, rows: np.ndarray) -> np.ndarray:
        """(N, 65, 65) psi from an unscaled coefficient block."""
        n = self.cfg.n_components
        stats = block[:, n:] if self.cfg.normalize else None
        return reconstruct(self.pca, block[:, :n], fr, rows, self.cfg.use_vacuum, stats=stats)

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block, scal = self.coefficients(fr, mask)
        return self.decode(block, fr, np.nonzero(mask)[0]), scal

    def r2_psi(self, fr: Frames, mask: np.ndarray) -> float:
        psi, _ = self.predict(fr, mask)
        return r2_psi(fr, np.nonzero(mask)[0], psi)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "net.pt")
        joblib.dump(self.pca, path / "pca.joblib")
        np.savez(
            path / "stats.npz",
            x_mean=self.x_mean,
            x_std=self.x_std,
            c_std=self.c_std,
            c_mean=self.c_mean,
            s_mean=self.s_mean,
            s_std=self.s_std,
            components=self.pca.components_,
            pca_mean=self.pca.mean_,
        )
        (path / "config.json").write_text(json.dumps({**asdict(self.cfg), "n_in": len(self.x_mean)}))

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> NeuralPredictor:
        raw = json.loads((path / "config.json").read_text())
        n_in = raw.pop("n_in")
        m = cls(NetConfig(**raw), device)
        st = np.load(path / "stats.npz")
        m.x_mean, m.x_std, m.c_std, m.s_mean, m.s_std = (
            st["x_mean"],
            st["x_std"],
            st["c_std"],
            st["s_mean"],
            st["s_std"],
        )
        m.c_mean = st["c_mean"] if "c_mean" in st else np.zeros_like(m.c_std)  # older models: zero-mean coefficients
        if (path / "pca.joblib").exists():
            m.pca = joblib.load(path / "pca.joblib")
        else:  # models saved before pca.joblib existed: rebuild the fitted state sklearn's inverse_transform needs
            m.pca = PCA(m.cfg.n_components)
            m.pca.components_, m.pca.mean_ = st["components"], st["pca_mean"]
            m.pca.n_components_ = m.cfg.n_components
            m.pca.explained_variance_ = np.ones(m.cfg.n_components)
        m.net = m._make_net(n_in)
        m.net.load_state_dict(torch.load(path / "net.pt", map_location=m.device))
        return m


class Ensemble:
    """Average of saved NeuralPredictors (psi and scalars); saved as a directory holding members.json."""

    def __init__(self, members: list[Path]) -> None:
        self.paths = members
        self.members = [NeuralPredictor.load(p) for p in members]

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi, scal = self.members[0].predict(fr, mask)
        psi, scal = psi.astype(np.float64), scal.astype(np.float64)
        for m in self.members[1:]:
            p, s = m.predict(fr, mask)
            psi += p
            scal += s
        return (psi / len(self.members)).astype(np.float32), (scal / len(self.members)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "members.json").write_text(json.dumps([str(p) for p in self.paths], indent=1))

    @classmethod
    def load(cls, path: Path) -> Ensemble:
        return cls([Path(p) for p in json.loads((path / "members.json").read_text())])


def smooth_in_time(x: np.ndarray, shot_index: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time within each shot (frames time-ordered per shot); sigma in frames, 0 leaves `x`
    unchanged."""
    if sigma <= 0:
        return x
    out = np.empty_like(x)
    for s, e in zip(*shot_bounds(shot_index), strict=True):
        gaussian_filter1d(x[s:e], sigma, axis=0, mode="nearest", output=out[s:e])
    return out


@dataclass
class Smoothed:
    """A predictor whose psi and scalars are smoothed along time within each shot; `mask` must select whole shots."""

    model: Predictor
    sigma: float

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi, scal = self.model.predict(fr, mask)
        shots = np.asarray(fr.shot_index)[mask]
        return smooth_in_time(psi, shots, self.sigma), smooth_in_time(scal, shots, self.sigma)
