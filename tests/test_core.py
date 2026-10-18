"""Green's functions, input alignment, context features, the coefficient loss, chunked targets, scorer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from fusioneq import targets
from fusioneq.cache import HISTORY_COLUMNS, Frames, ramp_frames
from fusioneq.data import interp_series, profile_on_efit, resample_profile
from fusioneq.greens import coil_basis, filament_flux, rectangle_flux
from fusioneq.model import (
    NetConfig,
    NeuralPredictor,
    TemporalNet,
    context_features,
    context_rows,
    max_windows,
    shot_windows,
)
from fusioneq.score import MASKS, Scorer, boundary_shape

GRID_R, GRID_Z = np.linspace(0.8, 2.5, 65), np.linspace(-1.6, 1.6, 65)


@pytest.fixture
def frames() -> SimpleNamespace:
    """A tiny stand-in for cache.Frames: 12 frames of 65x65 psi, vacuum part, and shot ids."""
    rng = np.random.default_rng(0)
    psi_vac = rng.normal(size=(12, 65, 65)).astype(np.float16)
    psi = (psi_vac.astype(np.float32) + rng.normal(scale=0.1, size=(12, 65, 65))).astype(np.float16)
    return SimpleNamespace(psi=psi, psi_vac=psi_vac, shot_index=np.repeat([0, 1, 2], 4))


class TestGreens:
    def test_filament_flux_is_positive_symmetric_and_decays(self):
        RR, ZZ = np.meshgrid(np.linspace(0.5, 2.5, 21), np.linspace(-1, 1, 21))
        psi = filament_flux(RR, ZZ, 1.5, 0.0)
        assert np.all(psi > 0) and np.allclose(psi, psi[::-1])
        along_z = filament_flux(np.full(3, 1.5), np.array([0.2, 0.8, 2.0]), 1.5, 0.0)
        assert along_z[0] > along_z[1] > along_z[2]

    def test_coil_basis_sums_conductors_joined_to_a_column(self):
        coils = {
            "coil_input_column": np.array(["magnetics_A", "magnetics_A", "magnetics_B"]),
            "coil_R": np.array([1.0, 1.1, 2.0]),
            "coil_Z": np.array([0.5, 0.5, -0.5]),
            "coil_width": np.full(3, 0.1),
            "coil_height": np.full(3, 0.1),
        }
        basis = coil_basis(GRID_R, GRID_Z, coils, ["magnetics_A", "magnetics_B", "magnetics_C"])
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        expected_a = (rectangle_flux(RR, ZZ, 1.0, 0.5, 0.1, 0.1) + rectangle_flux(RR, ZZ, 1.1, 0.5, 0.1, 0.1)) * 1e3
        assert np.allclose(basis[0], expected_a, rtol=1e-5)
        assert not np.allclose(basis[1], 0) and np.all(basis[2] == 0)  # a column with no geometry contributes nothing

    def test_ampere_turn_columns_share_their_total_across_elements(self):
        coils = {
            "coil_input_column": np.array(["magnetics_p2l_current"] * 2 + ["magnetics_sol_current"] * 2),
            "coil_R": np.array([1.0, 1.1, 0.2, 0.2]),
            "coil_Z": np.array([0.5, 0.5, -0.2, 0.2]),
            "coil_width": np.full(4, 0.05),
            "coil_height": np.full(4, 0.05),
        }
        basis = coil_basis(GRID_R, GRID_Z, coils, ["magnetics_p2l_current", "magnetics_sol_current"])
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        p2 = [rectangle_flux(RR, ZZ, r, 0.5, 0.05, 0.05) for r in (1.0, 1.1)]
        sol = [rectangle_flux(RR, ZZ, 0.2, z, 0.05, 0.05) for z in (-0.2, 0.2)]
        assert np.allclose(basis[0], (p2[0] + p2[1]) / 2 * 1e3, rtol=1e-5)  # kA-turn total split over its turns
        assert np.allclose(basis[1], (sol[0] + sol[1]) * 1e3, rtol=1e-5)  # per-turn kA summed over turns


class TestAlignment:
    def test_interp_ignores_nan_and_extrapolates_flat(self):
        v = interp_series(np.array([0.0, 5.0, 20.0]), np.array([0.0, 10.0, np.nan]), np.array([1.0, 3.0, 9.0]))
        assert np.allclose(v, [1.0, 2.0, 3.0])

    def test_nearest_profile_and_station_resampling(self):
        times = np.array([0.0, 10.0, 20.0])
        prof = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]])
        picked = profile_on_efit(times, prof, np.array([4.0, 16.0]))
        assert np.array_equal(picked[0], prof[0]) and np.array_equal(picked[1][:1], prof[2][:1])
        st = resample_profile(np.array([0.0, 1.0]), np.array([[2.0, 4.0]]), np.array([0.5, 5.0]))
        assert st[0, 0] == pytest.approx(3.0) and st[0, 1] == 0.0

    def test_windowed_profile_averages_finite_samples_and_falls_back_to_nearest(self):
        times = np.array([0.0, 4.0, 8.0, 30.0])
        prof = np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0], [9.0, 9.0]])
        out = profile_on_efit(times, prof, np.array([4.0, 20.0]), window=10.0)
        # frame at 4: samples 0,4,8 in window; channel 1 skips the NaN. frame at 20: none in [15,25] -> nearest (30)
        assert np.allclose(out, [[3.0, 6.0], [9.0, 9.0]])
        assert np.array_equal(profile_on_efit(times, prof, np.array([4.0]), 0.0), prof[[1]])

    def test_samples_out_of_time_order_are_dropped_before_matching(self):
        times = np.array([0.0, 10.0, 10.0, 5.0, 20.0])  # a repeated pad time, then one sample out of order
        prof = np.array([[1.0], [2.0], [99.0], [99.0], [3.0]])
        assert profile_on_efit(times, prof, np.array([6.0, 11.0, 19.0]), window=4.0).ravel().tolist() == [2.0, 2.0, 3.0]

    def test_context_features_respect_shot_boundaries(self):
        out = context_features(np.arange(6, dtype=np.float32)[:, None], np.array([0, 0, 0, 1, 1, 1]), 1)
        assert out[0].tolist() == [0.0, 0.0, 1.0]  # previous frame edge-padded within shot 0
        assert out[2].tolist() == [2.0, 1.0, 2.0]  # last frame of shot 0 does not see shot 1
        assert out[3].tolist() == [3.0, 3.0, 4.0]  # first frame of shot 1 does not see shot 0

    def test_context_rows_match_the_full_context_build(self):
        rng = np.random.default_rng(0)
        n = 30
        fr = Frames(
            currents=rng.normal(size=(n, 3)),
            ip=rng.normal(size=n),
            thomson=rng.normal(size=(n, 4)),
            psi_vac=None,
            shot_index=np.repeat([0, 1, 2], 10),
            efit_times=None,
            grid_R=None,
            grid_Z=None,
            shots=[],
        )
        rows = np.array([0, 4, 9, 10, 19, 29])
        full = context_features(fr.features(), fr.shot_index, 2)[rows]
        assert np.array_equal(context_rows(fr, rows, NetConfig(context=2)), full)


class TestRampFrames:
    def test_marks_shot_edges_and_low_current_per_shot(self):
        shot_index = np.repeat([0, 1], 20)
        times = np.tile(np.arange(20.0), 2)
        ip = np.r_[np.full(20, 1000.0), np.full(20, 500.0)]
        ip[30] = 300.0  # a dip below 80% of shot 1's own peak
        ramp = ramp_frames(shot_index, times, ip)
        assert ramp[[0, 1, 18, 19, 20, 39, 30]].all()  # each shot's first / last 10%, and the dip
        assert not ramp[[5, 10, 25]].any()  # flat-top, judged against each shot's own peak


class TestTemporal:
    def test_shot_windows_tile_every_frame_once_and_never_cross_shots(self):
        shots = np.repeat([0, 1, 2], [5, 12, 3])
        idx, valid = shot_windows(shots, 4, np.random.default_rng(0))
        assert idx.shape == valid.shape and idx.shape[1] == 4
        assert sorted(idx[valid].tolist()) == list(range(len(shots)))  # each frame scored exactly once per epoch
        assert all(len(set(shots[w])) == 1 for w in idx)  # padding repeats the shot's own edge frame
        counts = {len(shot_windows(shots, 4, np.random.default_rng(seed))[0]) for seed in range(20)}
        assert max(counts) <= max_windows(shots, 4)  # the LR schedule's step budget covers every epoch

    def test_temporal_net_sees_both_directions_within_its_receptive_field(self):
        torch.manual_seed(0)
        net = TemporalNet(n_in=3, n_out=2, hidden=8, layers=1, dropout=0.0, temporal_layers=2).eval()
        x = torch.randn(1, 40, 3)
        bumped = x.clone()
        bumped[0, 20] += 5.0
        changed = (net(bumped) - net(x)).abs().sum(-1)[0] > 1e-6
        reach = 2 * (1 + 2)  # kernel 5: +-2 frames per block, dilations 1 and 2
        assert changed[20 - reach] and changed[20 + reach]  # past and future frames both respond
        assert not changed[: 20 - reach].any() and not changed[20 + reach + 1 :].any()


class TestHistory:
    def test_history_averages_the_past_within_each_shot_only(self):
        n = 60
        ip = np.r_[np.arange(40, dtype=float), np.full(20, 100.0)] * -1  # shot 0 ramps, shot 1 is flat; sign ignored
        thomson = np.zeros((n, 40))
        thomson[:, 3] = np.r_[np.zeros(40), np.full(20, 2.0)]  # core Te peak
        fr = Frames(
            currents=None,
            ip=ip,
            thomson=thomson,
            psi_vac=None,
            shot_index=np.repeat([0, 1], [40, 20]),
            efit_times=np.r_[np.arange(40), np.arange(20)] * 20.0,
            grid_R=None,
            grid_Z=None,
            shots=[],
        )
        h = fr.history
        assert (
            h.shape == (n, len(HISTORY_COLUMNS)) and h[39, 0] == pytest.approx(0.78) and h[40, 0] == 0.0
        )  # time in shot [s]
        assert h[39, 1] == pytest.approx(np.mean(np.arange(30, 40)))  # |Ip| over the last 10 frames
        assert h[5, 2] == pytest.approx(np.mean(np.arange(6)))  # fewer frames at the shot's start
        assert h[39, 4] == pytest.approx(5.0) and h[4, 4] == pytest.approx(4.0)  # ramp over 5 frames, clamped
        assert np.allclose(h[40:, 1:4], 100.0) and np.allclose(h[40:, 4], 0.0)  # shot 1 never sees shot 0
        assert np.allclose(h[40:, 5:], 2.0) and np.allclose(h[:40, 5:], 0.0)


class TestCoefLoss:
    def test_pixel_weights_make_the_loss_psi_mse(self):
        rng = np.random.default_rng(0)
        c_std = np.array([4.0, 1.0, 0.25], np.float32)
        model = NeuralPredictor(NetConfig(n_components=3, coef_loss="pixel"), device="cpu")
        model.c_std = c_std
        w = model.coef_weights("cpu").numpy()
        err_std = rng.normal(size=(64, 3)).astype(np.float32)  # error in standardised-coefficient space
        # an orthonormal PCA basis makes psi MSE proportional to the unstandardised coefficient MSE
        assert np.allclose((err_std**2 * w).mean(), ((err_std * c_std) ** 2).mean() / (c_std**2).mean(), rtol=1e-5)
        model.cfg.coef_loss = "std"
        assert np.allclose(model.coef_weights("cpu").numpy(), 1.0)
        model.cfg.coef_loss = "mix"
        assert np.allclose(model.coef_weights("cpu").numpy(), 0.5 + 0.5 * w)

    def test_region_metric_measures_psi_error_inside_the_envelope_only(self):
        rng = np.random.default_rng(0)
        model = NeuralPredictor(NetConfig(n_components=4), device="cpu")
        model.pca = SimpleNamespace(components_=np.linalg.qr(rng.normal(size=(65 * 65, 4)))[0].T)
        model.c_std = np.array([3.0, 1.0, 0.5, 0.1])
        M = model.region_metric("cpu").numpy().astype(np.float64)
        inside = np.load(MASKS["DIII-D"])["mask_coarse"].astype(bool).ravel()
        d = rng.normal(size=(5, 4))
        err_inside = (((d * model.c_std) @ model.pca.components_)[:, inside] ** 2).sum(1)
        ratio = np.einsum("bi,ij,bj->b", d, M, d) / err_inside
        assert np.trace(M) == pytest.approx(4) and np.allclose(ratio, ratio[0], rtol=1e-4)


class TestTargets:
    def test_chunked_paths_match_the_unchunked_computation(self, frames, monkeypatch):
        monkeypatch.setattr(targets, "CHUNK", 5)  # force several uneven chunks over 12 frames
        rows = np.arange(12)
        pca = PCA(4, random_state=0).fit((frames.psi - frames.psi_vac).astype(np.float32).reshape(12, -1))
        coef = targets.transform_all(pca, frames, rows, use_vacuum=True)
        direct = pca.transform((frames.psi - frames.psi_vac).astype(np.float32).reshape(12, -1))
        assert np.allclose(coef, direct, atol=1e-4)
        psi = targets.reconstruct(pca, coef, frames, rows, use_vacuum=True)
        gt = frames.psi.astype(np.float32)
        assert targets.r2_psi(frames, rows, psi) == pytest.approx(
            1 - ((psi - gt) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
        )
        assert targets.r2_psi(frames, rows, gt) == pytest.approx(1.0)

    def test_normalized_target_is_unit_shape_and_inverts_exactly(self, frames):
        rows = np.arange(12)
        flat = np.concatenate([t for _, t in targets.residual_chunks(frames, rows, True, normalize=True)])
        inside = flat[:, targets.envelope()]
        assert np.allclose(inside.mean(1), 0, atol=1e-5) and np.allclose(inside.std(1), 1, atol=1e-4)
        pca = PCA(12).fit(flat)  # full rank on 12 frames: the round trip must be exact
        coef, stats = targets.transform_shape(pca, frames, rows)
        assert np.allclose(coef, targets.transform_all(pca, frames, rows, True, normalize=True), atol=1e-5)
        psi = targets.reconstruct(pca, coef, frames, rows, True, stats=stats)
        assert np.allclose(psi, frames.psi.astype(np.float32), atol=1e-3)


class TestScorer:
    def test_perfect_and_zero_predictions(self):
        sc = Scorer("DIII-D")
        RR, ZZ = np.meshgrid(sc.R, sc.Z)
        psi = np.stack([-(np.exp(-(((RR - 1.7) / 0.5) ** 2 + ((ZZ - z0) / 0.9) ** 2))) for z0 in (0.0, 0.05, -0.05)])
        gt = {"psi": psi.astype(np.float32), "q95": np.array([4.0, 4.5, 5.0]), "betaN": np.array([1.0, 1.2, 1.4])}
        perfect = sc.score([gt], [{"psirz": gt["psi"], "q95": gt["q95"], "betaN": gt["betaN"]}], workers=1)
        assert perfect["S"] == pytest.approx(1.0, abs=1e-4)
        zeros = sc.score([gt], [{"psirz": np.zeros_like(psi), "q95": np.zeros(3), "betaN": np.zeros(3)}], workers=1)
        assert zeros["S"] == pytest.approx(0.0, abs=1e-4)

    def test_boundary_shape_of_a_d_shaped_plasma(self):
        t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        r, z = 1.7 + 0.6 * np.cos(t + 0.4 * np.sin(t)), 1.8 * 0.6 * np.sin(t)
        r0, a, kappa, delta = boundary_shape(r, z)
        assert np.allclose([a, kappa], [0.6, 1.8], atol=1e-3)
        assert r0 == pytest.approx(1.7, abs=1e-3) and delta == pytest.approx(np.sin(0.4), abs=1e-2)
