"""Lever 1b: the vacuum-field shape map, the per-frame shape ladder and its label-free gate."""

from __future__ import annotations

import numpy as np
import pytest

from fusioneq.greens import MU0
from fusioneq.mast_shape import Deriver, ShapeLadder, cap_coils, choose, gate, summary
from fusioneq.physics import perturb_placement
from fusioneq.plasma import PlasmaShape, current_density, ellipse_rho2, shaped_rho2
from fusioneq.scalars import LI_HALF, shafranov_drive
from fusioneq.score import boundary_shape
from fusioneq.shape_map import DELTA_MAX, FEATURES, ShapeMap, ShiftModel, features, moments, placement_shape

GRID_R, GRID_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
RR, ZZ = np.meshgrid(GRID_R, GRID_Z)


def poly_flux(r0: float, a: float, c: dict[tuple[int, int], float]) -> np.ndarray:
    x, y = (RR - r0) / a, ZZ / a
    return sum(v * x**i * y**j for (i, j), v in c.items())


def bowl(r0: float, z0: float, kappa: float) -> np.ndarray:
    """Flux whose axis is a minimum at (r0, z0) with elliptical surfaces (MAST orientation: axis_sign -1)."""
    return ((RR - r0) / 0.5) ** 2 + ((ZZ - z0) / (0.5 * kappa)) ** 2


class TestShapeMap:
    COEF = {(0, 0): 0.3, (1, 0): 0.8, (0, 1): -0.1, (2, 0): 0.05, (1, 1): 0.02, (0, 2): -0.04, (2, 1): 0.01}

    def test_moments_recover_a_cubic_and_features_ignore_sign_and_scale(self):
        r0, a = np.array([0.9, 1.0]), np.array([0.5, 0.45])
        psi = np.stack([poly_flux(r, s, self.COEF) for r, s in zip(r0, a, strict=True)])
        c = moments(psi, GRID_R, GRID_Z, r0, a)
        assert c[:, 1] == pytest.approx([0.8, 0.8], rel=1e-3) and c[:, 2] == pytest.approx([-0.1, -0.1], abs=1e-3)
        f = features(psi, GRID_R, GRID_Z, r0, a, np.array([500.0, 500.0]))
        assert f[:, FEATURES.index("tilt")] == pytest.approx([-0.125, -0.125], rel=1e-2)
        assert f[0, FEATURES.index("decay_index")] == pytest.approx(1 - 2 * 1.8 * 0.05 / 0.8, rel=1e-2)
        g = features(-3.0 * psi, GRID_R, GRID_Z, r0, a, np.array([500.0, 500.0]))
        shape_only = [FEATURES.index(n) for n in FEATURES if n != "lambda_v"]
        assert np.allclose(f[:, shape_only], g[:, shape_only], rtol=1e-6)

    def test_fit_recovers_a_linear_map_clips_and_roundtrips(self, tmp_path):
        rng = np.random.default_rng(0)

        def truth(X: np.ndarray) -> np.ndarray:
            return np.c_[0.05 * X[:, 0], 1.7 + 0.2 * X[:, 1], 0.4 + 0.1 * X[:, 3], 0.4 - 0.1 * X[:, 4]]

        X = rng.uniform(-1, 1, size=(2000, len(FEATURES)))
        m = ShapeMap.fit(X, truth(X), lam=1e-6)
        inside = 0.9 * X[:50]  # within the feature and target clip ranges
        assert np.allclose(m.predict(inside), truth(inside), atol=1e-3)
        far = np.full((1, len(FEATURES)), 50.0)
        assert np.all(m.predict(far) <= np.array(m.y_hi) + 1e-12)
        assert m.predict(np.full((1, len(FEATURES)), np.nan)) == pytest.approx(np.array([m.intercept]))
        m.save(tmp_path / "m.json")
        assert np.allclose(ShapeMap.load(tmp_path / "m.json").predict(X[:5]), m.predict(X[:5]))


class TestPlasmaShape:
    def test_zero_triangularity_is_the_ellipse_and_positive_moves_the_top_inward(self):
        assert np.allclose(
            shaped_rho2(RR, ZZ, 0.9, 0.5, 0.0, PlasmaShape(1.8)), ellipse_rho2(RR, ZZ, 0.9, 0.5, 0.0, 1.8)
        )
        j0 = current_density(GRID_R, GRID_Z, 0.9, 0.5, 0.0, PlasmaShape(1.8, 1.0))
        jd = current_density(GRID_R, GRID_Z, 0.9, 0.5, 0.0, PlasmaShape(1.8, 1.0, delta_top=0.5))
        top = ZZ > 0.5

        def r_mean(j: np.ndarray, sel: np.ndarray) -> float:
            return float((j * RR)[sel].sum() / j[sel].sum())

        assert r_mean(jd, top) < r_mean(j0, top) - 0.02
        assert r_mean(jd, ZZ < -0.5) == pytest.approx(r_mean(j0, ZZ < -0.5))


class TestPlacementScalars:
    """Geometry with no extraction: q95 / betaN shape rows, the Shafranov shift, the replay's placement error."""

    def test_placement_shape_matches_the_boundary_definition_of_a_d_shape(self):
        r0, a, kappa, d_top, d_bot = 0.9, 0.5, 1.8, 0.4, 0.2
        t = np.linspace(0, 2 * np.pi, 4001)
        delta = np.where(np.sin(t) >= 0, d_top, d_bot)
        r, z = r0 + a * np.cos(t + np.arcsin(delta) * np.sin(t)), kappa * a * np.sin(t)
        pred = np.array([[0.0, kappa, d_top, d_bot]])
        got = placement_shape(np.array([[r0, a]]), pred)[0]
        assert got == pytest.approx(boundary_shape(r, z), abs=2e-3)
        assert placement_shape(np.array([[r0, a]]), pred, use_tri=False)[0, 3] == 0.0

    def test_shift_model_recovers_a_line_clips_and_fills_missing_drive(self, tmp_path):
        x = np.linspace(0.1, 0.5, 50)
        model = ShiftModel.fit(np.r_[x, np.nan], np.r_[0.02 + 0.3 * x, 0.1])
        assert (model.c0, model.c1) == pytest.approx((0.02, 0.3))
        assert model.predict(np.array([-1.0, 10.0, np.nan, 0.3])) == pytest.approx([0.0, DELTA_MAX, 0.11, 0.11])
        model.save(tmp_path / "s.json")
        assert ShiftModel.load(tmp_path / "s.json") == model

    def test_shafranov_drive_is_eps_times_beta_p_plus_half_li(self):
        r0, a, kappa = np.array([1.7]), np.array([0.6]), np.array([1.0])
        bp = MU0 * 1e6 / (2 * np.pi * 0.6)  # 1 MA, circular
        pe = np.array([0.0, bp**2 / (2 * MU0)])  # beta_p 0 and 1
        got = shafranov_drive(np.r_[r0, r0], np.r_[a, a], np.r_[kappa, kappa], pe, np.array([1e3, -1e3]))
        assert got == pytest.approx(0.6 / 1.7 * np.array([LI_HALF, 1 + LI_HALF]))

    def test_perturbation_is_a_per_shot_shift_and_bias_plus_frame_jitter(self):
        shots = np.repeat(np.arange(400), 50)
        geom = np.c_[np.full(len(shots), 1.7), np.full(len(shots), 0.6), np.zeros(len(shots))]
        out = perturb_placement(geom, shots, seed=3)
        assert np.array_equal(out, perturb_placement(geom, shots, seed=3)) and geom[0, 0] == 1.7
        dr = (out[:, 0] - 1.7).reshape(400, 50)
        assert np.allclose(dr, dr[:, :1]) and dr[:, 0].std() == pytest.approx(0.02, rel=0.15)
        ratio = (out[:, 1] / 0.6).reshape(400, 50)
        assert np.log(ratio).mean(1).std() == pytest.approx(0.08, rel=0.15)
        assert np.log(ratio).std(1).mean() == pytest.approx(0.04, rel=0.15)
        assert np.array_equal(out[:, 2], geom[:, 2])


class TestLadderGate:
    def test_derived_shape_follows_the_flux(self):
        d = Deriver(GRID_R, GRID_Z, "MAST", -1.0).rows(np.stack([bowl(1.0, 0.1, 1.5), bowl(1.0, 0.0, 2.0)]))
        assert d[:, 7].tolist() == [1, 1]
        assert d[:, 1] == pytest.approx([0.1, 0.0], abs=0.02)
        assert d[0, 2] < d[1, 2]

    def test_gate_needs_closed_lcfs_axis_inside_and_kappa_in_band(self):
        geom = np.array([[1.0, 0.5, 0.0]] * 4)
        derived = np.zeros((4, 8))
        derived[:, :3] = [[1.0, 0.0, 1.8], [1.0, 0.0, 3.4], [0.5, 1.7, 1.8], [1.0, 0.0, 1.8]]
        derived[:, 7] = [1, 1, 1, 0]
        assert gate(derived, geom, 1.6).tolist() == [True, False, False, False]

    def test_choose_takes_the_nearest_passing_candidate(self):
        kappa = np.array([[1.5, 1.9, 1.5], [1.8, 2.3, 1.7]])
        ok = np.array([[True, True, False], [True, False, False]])
        assert choose(kappa, ok, np.array([1.75, 2.3, 1.7])).tolist() == [1, 0, -1]

    def test_cap_coils_only_binds_where_a_coil_outruns_the_plasma(self):
        psi = np.stack([bowl(1.0, 0.0, 1.6), bowl(1.0, 0.0, 1.6)])
        guard = (np.abs(RR - 0.5) < 0.05) & (np.abs(ZZ - 1.7) < 0.1)
        psi[1][guard] = psi[1].min() - 1.0  # a coil spike deeper than the plasma axis (axis_sign -1: minimum)
        region = ((RR - 1.0) / 0.5) ** 2 + (ZZ / 0.8) ** 2 <= 1
        out = cap_coils(psi, guard, np.stack([region, region]), -1.0)
        assert np.allclose(out[0], psi[0], atol=1e-6)
        assert out[1][guard].min() > psi[1][region].min()
        assert np.allclose(out[1][~guard], psi[1][~guard], atol=1e-6)

    def test_ladder_picks_the_target_kappa_and_feeds_back_the_axis(self):
        def plasma(kappa: float, place: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
            # the O-point sits 3 cm above and 2 cm inboard of the ellipse centre
            return np.stack([bowl(r - 0.02, z + 0.03, kappa) for r, _, z in place])

        ladder = ShapeLadder(Deriver(GRID_R, GRID_Z, "MAST", -1.0), np.zeros(RR.shape, bool), workers=1)
        geom = np.array([[1.0, 0.5, 0.0]] * 2)
        z_target, r_target = np.array([0.05, 0.05]), np.array([1.06, 1.0])
        psi, rec = ladder.run(np.zeros((2, *RR.shape)), plasma, geom, z_target, np.array([1.3, 2.0]), r_target)
        assert rec["kappa_e"].tolist() == [1.3, 2.0]
        assert rec["z_refined"].all() and rec["z0"] == pytest.approx([0.02, 0.02], abs=0.01)
        assert rec["r_refined"].all() and rec["place"][:, 0] == pytest.approx([1.08, 1.02], abs=0.01)
        assert rec["derived"][:, :2] == pytest.approx(np.c_[r_target, z_target], abs=0.02)
        assert psi.dtype == np.float32 and np.allclose(psi, psi.astype(np.float16))
        _, plain = ladder.run(np.zeros((2, *RR.shape)), plasma, geom, z_target, np.array([1.3, 2.0]))
        assert "r_refined" not in plain and plain["place"][:, 0].tolist() == [1.0, 1.0]


def test_summary_flags_and_within_shot_variation():
    derived = np.zeros((6, 8))
    derived[:, 2] = [1.6, 1.7, 1.8, 2.0, 2.0, 3.0]
    derived[:, 7] = [1, 1, 1, 1, 1, 0]
    derived[:, :2] = [1.0, 0.0]
    s = summary(derived, np.array([0, 0, 0, 1, 1, 1]), np.array([[1.0, 0.5, 0.0]] * 6))
    assert s["flags"] == pytest.approx({"lcfs_fail": 1 / 6, "kappa_out": 1 / 6, "axis_off": 0.0})
    assert s["kappa"]["step"] == pytest.approx(0.1)
    assert 0 < s["kappa"]["within"] < 1
