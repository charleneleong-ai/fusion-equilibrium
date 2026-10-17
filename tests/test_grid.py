"""Grid shape model: the Grad-Shafranov operator, fitting a feature-dependent shape (U-Net, FNO), the Fourier layer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fusioneq import grid
from fusioneq.grid import PAD, GridConfig, GridPredictor, SpectralConv, gs_operator, make_net, shape_maps
from fusioneq.targets import envelope

RR, ZZ = np.meshgrid(np.linspace(-1, 1, 65), np.linspace(-1, 1, 65))
GRID_R, GRID_Z = np.linspace(1.0, 2.5, 65), np.linspace(-1.5, 1.5, 65)
R_PHYS, Z_PHYS = np.meshgrid(GRID_R, GRID_Z)
# (psi, analytic Delta* psi): vacuum solutions give zero; R^4 gives 8 R^2
DELTA_STAR = {
    "R^2": (R_PHYS**2, np.zeros_like(R_PHYS)),
    "Z": (Z_PHYS, np.zeros_like(R_PHYS)),
    "R^2 Z": (R_PHYS**2 * Z_PHYS, np.zeros_like(R_PHYS)),
    "R^4": (R_PHYS**4, 8 * R_PHYS**2),
}
# (arch overrides, Grad-Shafranov loss weight) for the fit test; the FNO needs width 8 to clear the bar in 40 epochs
FITS = {
    "unet": ({}, 0.0),
    "unet-gs": ({}, 1.0),
    "fno": ({"arch": "fno", "channels": 8, "modes": 8, "layers": 2}, 0.0),
}


class TinyFrames:
    """Frames stand-in: one feature per frame that moves a Gaussian bump in R; a smooth vacuum field."""

    def __init__(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(-0.4, 0.4, n).astype(np.float32)
        bump = np.exp(-((RR[None] - self.x[:, None, None]) ** 2 + ZZ[None] ** 2) / 0.1)
        self.psi_vac = np.broadcast_to((RR + 0.5 * ZZ).astype(np.float16), (n, 65, 65))
        self.psi = (self.psi_vac + 2.0 * bump).astype(np.float16)
        self.shot_index = np.repeat(np.arange(n // 4), 4)
        self.grid_R, self.grid_Z = GRID_R, GRID_Z

    def features(self, rows: np.ndarray, thomson_hr: bool = False) -> np.ndarray:
        return self.x[rows, None]


class Logged:
    """A Run stand-in that keeps every logged row."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def log(self, metrics: dict[str, float]) -> None:
        self.rows.append(metrics)


@pytest.fixture
def cfg() -> GridConfig:
    return GridConfig(channels=4, context=0, film_hidden=16, epochs=40, batch=16, stride=1, amp=False, lr=3e-3)


class TestGrid:
    @pytest.mark.parametrize(("psi", "delta"), DELTA_STAR.values(), ids=DELTA_STAR)
    def test_gs_operator_matches_the_analytic_delta_star(self, psi, delta):
        dR, dZ = GRID_R[1] - GRID_R[0], GRID_Z[1] - GRID_Z[0]
        got = gs_operator(torch.from_numpy(psi[None]), torch.from_numpy(GRID_R), dR, dZ)[0].numpy()
        assert np.allclose(got, delta[1:-1, 1:-1] * dR**2, atol=1e-3 * dR**2 * 8 * GRID_R.max() ** 2)

    @pytest.mark.parametrize(("arch", "gs_weight"), FITS.values(), ids=FITS)
    def test_fit_learns_a_feature_dependent_shape_and_round_trips(self, cfg, arch, gs_weight, tmp_path):
        fr = TinyFrames(64, seed=0)
        rows = np.arange(64)
        train, val = rows < 48, rows >= 48
        cfg = replace(cfg, gs_weight=gs_weight, **arch)
        m = GridPredictor(cfg, device="cpu").fit(fr, train, val, checkpoint=tmp_path / "grid")
        pred = m.predict_shape(fr, val)
        shapes = shape_maps(fr.psi, rows, np.float32).reshape(64, -1)[:, envelope()]
        mse = np.mean((pred.reshape(16, -1)[:, envelope()] - shapes[val]) ** 2)
        assert mse < 0.25 * np.mean((shapes[train].mean(0) - shapes[val]) ** 2)  # well below the mean training shape
        again = GridPredictor.load(tmp_path / "grid", device="cpu")  # the best epoch, checkpointed during training
        assert np.allclose(again.predict_shape(fr, val), pred, atol=1e-5)
        drift = np.abs(m.predict_shape(fr, val, bf16=True) - pred).max()  # what a GPU build under autocast ships
        assert 0 < drift < 0.05 * np.abs(pred).max()

    def test_reports_progress_within_an_epoch_with_a_rising_position(self, cfg, monkeypatch):
        monkeypatch.setattr(grid, "PROGRESS_EVERY_S", 0)
        cfg.epochs, run = 2, Logged()
        fr, rows = TinyFrames(64, seed=0), np.arange(64)
        GridPredictor(cfg, device="cpu").fit(fr, rows < 48, rows >= 48, run=run)
        progress = [r["epoch_progress"] for r in run.rows if "epoch_progress" in r]
        assert len(progress) == 2 * 3 and progress == sorted(progress) and progress[-1] == 2.0  # 48 frames / batch 16


def band_limited(n: int, channels: int, seed: int) -> torch.Tensor:
    """(1, channels, n, n) sum of random Fourier modes up to frequency 5, sampled on a periodic n x n grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    h = np.zeros((channels, n, n))
    for kz, kr in rng.integers(0, 6, (6, 2)):
        amp, phase = rng.normal(size=(channels, 1, 1)), rng.uniform(0, 2 * np.pi)
        h += amp * np.cos(2 * np.pi * (kz * t[:, None] + kr * t[None, :]) + phase)
    return torch.from_numpy(h[None].astype(np.float32))


class TestFourierLayer:
    def test_spectral_conv_is_resolution_consistent_on_the_padded_grid(self):
        torch.manual_seed(0)
        layer = SpectralConv(channels=3, modes=8)
        with torch.no_grad():
            coarse, fine = layer(band_limited(PAD, 3, seed=1)), layer(band_limited(2 * PAD, 3, seed=1))
        assert coarse.shape == (1, 3, PAD, PAD) and coarse.abs().max() > 1e-3
        assert torch.allclose(fine[..., ::2, ::2], coarse, atol=1e-5 * coarse.abs().max())

    @pytest.mark.parametrize("change", [{"arch": "vit"}, {"arch": "fno", "modes": PAD // 2 + 1}])
    def test_make_net_rejects_an_unknown_arch_or_aliasing_modes(self, cfg, change):
        with pytest.raises(ValueError):
            make_net(replace(cfg, **change), n_in=1)
