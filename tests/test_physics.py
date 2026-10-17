"""MAST physics transfer: plasma current placement, calibration, the Grad-Shafranov solve, q95 / betaN."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusioneq.agnostic import MachineFrame
from fusioneq.equilibrium import GradShafranov, Profile
from fusioneq.greens import filament_flux
from fusioneq.mast_geometry import BasisDiff, SolenoidSpec, freegs_basis
from fusioneq.physics import (
    MAST_AXIS_SIGN,
    Calibration,
    DemoScalarFit,
    MastCache,
    SolenoidOffset,
    fill_from_shot,
    plausible_boundary,
)
from fusioneq.plasma import PlasmaShape, current_density, density_extent
from fusioneq.plasma_region import Region, coil_cells, inside_polygon, line_of_sight
from fusioneq.scalars import ScalarCalibration, raw_scalars, tracking_stats, volume_average
from fusioneq.score import lcfs_shapes
from fusioneq.vessel import PassiveSet, annular_plate, cylinder_shell, join

GRID_R, GRID_Z = np.linspace(0.8, 2.5, 65), np.linspace(-1.6, 1.6, 65)
MAST_R, MAST_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
P2_COILS = {  # two P2-like rectangles near the centre column, top and bottom
    "coil_R": np.array([0.5, 0.5]),
    "coil_Z": np.array([1.75, -1.75]),
    "coil_width": np.array([0.08, 0.08]),
    "coil_height": np.array([0.2, 0.2]),
}


def mast_psi_with_coil_peak() -> np.ndarray:
    """MAST-oriented flux: a plasma at (0.9, 0) and a stronger, narrow flux peak inside the upper P2 coil."""
    RR, ZZ = np.meshgrid(MAST_R, MAST_Z)
    plasma = np.exp(-(((RR - 0.9) / 0.5) ** 2 + (ZZ / 0.9) ** 2))
    coil = 2.0 * np.exp(-(((RR - 0.5) / 0.08) ** 2 + ((ZZ - 1.75) / 0.12) ** 2))
    return MAST_AXIS_SIGN * (plasma + coil)


class TestPlasma:
    def test_density_extent_and_current_density(self):
        R = np.linspace(0.2, 1.5, 27)
        ne = np.exp(-(((R - 0.85) / 0.35) ** 2)) * 3e19
        ne[3] = np.nan
        r_in, r_out = density_extent(R, ne, frac=0.25)
        assert r_in == pytest.approx(0.85 - 0.35 * np.sqrt(np.log(4)), abs=0.06)
        assert r_out == pytest.approx(0.85 + 0.35 * np.sqrt(np.log(4)), abs=0.06)
        j = current_density(GRID_R, GRID_Z, 1.7, 0.6, 0.0, PlasmaShape(kappa=1.8))
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        assert j.sum() == pytest.approx(1.0) and np.all(j >= 0)
        assert np.all(j[((RR - 1.7) / 0.6) ** 2 + (ZZ / 1.08) ** 2 > 1] == 0)  # nothing outside the ellipse
        assert RR.ravel()[j.argmax()] == pytest.approx(1.7, abs=0.03)
        assert density_extent(R, np.full_like(R, np.nan)) is None


class TestPhysics:
    def test_calibration_recovers_scale_turns_and_offset(self):
        rng = np.random.default_rng(0)
        vac, ecoil, plasma = rng.normal(size=(3, 50, 65, 65)).astype(np.float32)
        truth = Calibration(scale=-0.9, plasma_ratio=0.8, offset=0.02, ecoil_turns=120.0)
        fit = Calibration.fit(vac, ecoil, plasma, truth.apply(vac, plasma, ecoil))
        assert np.allclose(list(vars(fit).values()), list(vars(truth).values()), rtol=1e-3)

    def test_solenoid_offset_recovers_a_and_b_and_corrects_to_the_truth(self):
        rng = np.random.default_rng(0)
        psi, i_sol = rng.normal(size=(20, 4, 4)), rng.uniform(-40, 20, 20)
        truth = MAST_AXIS_SIGN * psi + (-0.02 - 0.0007 * i_sol)[:, None, None]
        off = SolenoidOffset.fit(psi, truth, i_sol)
        assert np.allclose([off.a, off.b], [-0.02, -0.0007])
        assert np.allclose(MAST_AXIS_SIGN * off.apply(psi, i_sol), truth, atol=1e-6)

    def test_demo_scalar_fit_recovers_each_line_and_skips_missing_labels(self):
        pred = np.random.default_rng(0).uniform(1, 9, size=(30, 2))
        truth = np.c_[0.7 + 1.05 * pred[:, 0], 0.55 + 0.5 * pred[:, 1]]
        truth[3, 1] = np.nan
        fit = DemoScalarFit.fit(pred, truth)
        assert np.allclose(list(vars(fit).values()), [0.7, 1.05, 0.55, 0.5])
        assert np.allclose(fit.apply(pred)[:, 0], truth[:, 0])


def small_solver() -> GradShafranov:
    grid = np.linspace(0.5, 1.5, 17)
    envelope = np.zeros((17, 17), bool)
    envelope[2:-2, 2:-2] = True
    G = np.random.default_rng(0).uniform(0, 1e-7, size=(289, 289))
    return GradShafranov(grid, grid, envelope, (G + G.T) / 2, "cpu")


class TestEquilibrium:
    def test_zero_iterations_is_the_superposition_in_either_current_sign(self):
        gs, rng = small_solver(), np.random.default_rng(1)
        vac, j0 = rng.normal(size=(2, 289)), rng.dirichlet(np.ones(289), size=2)
        ip = np.array([2e5, -3e5])
        psi, _ = gs.solve(vac, ip, j0, Profile(), iters=0)
        assert np.allclose(psi, vac + (j0 * ip[:, None]) @ gs.Gt.numpy(), rtol=1e-5, atol=1e-6)

    def test_flood_fill_keeps_only_the_region_connected_to_the_seed(self):
        gs = small_solver()
        allowed = np.zeros((1, 17, 17), bool)
        allowed[0, 3:6, 3:6] = allowed[0, 10:13, 10:13] = True
        region = gs.connected(torch.as_tensor(allowed.reshape(1, -1)), torch.tensor([[4 * 17 + 4]]))
        assert region.sum() == 9 and region.view(17, 17)[11, 11] == 0

    def test_recentre_moves_the_current_centroid_to_the_target_and_keeps_ip(self):
        gs = small_solver()
        j = torch.zeros(1, 17, 17)
        j[0, 5:8, 5:8] = 1.0
        target, ip = torch.tensor([[1.1, 0.95]]), torch.tensor([[2e5]])
        moved = gs.recentre(j.view(1, -1), target, ip)
        assert torch.allclose(gs.centroid(moved), target, atol=1e-3) and moved.sum().item() == pytest.approx(2e5)

    def test_xpoint_flux_finds_an_analytic_saddle(self):
        gs = small_solver()
        R, Z = np.meshgrid(gs.grid_R, gs.grid_Z)
        phi = torch.as_tensor(((Z - 1.0) ** 2 - (R - 1.0) ** 2).reshape(1, -1), dtype=torch.float32)
        assert gs.xpoint_flux(phi, torch.tensor([[10.0]])).item() == pytest.approx(0.0, abs=1e-6)


class TestScalars:
    def test_calibration_recovers_constants_caps_q95_and_fills_missing(self):
        raw = np.random.default_rng(0).uniform(1, 5, size=(500, 2))
        cal = ScalarCalibration.fit(raw, raw * [0.8, 1.5])
        assert (cal.c_q, cal.c_b) == pytest.approx((0.8, 1.5))
        out = cal.apply(np.array([[100.0, 1.0], [np.nan, 1.0]]))
        assert out[0, 0] == pytest.approx(cal.q95_max) and out[1, 0] == pytest.approx(cal.fallback[0])

    def test_eps_cap_only_changes_high_eps_q95(self):
        shape = np.array([[0.85, 0.40, 1.9, 0.4], [0.75, 0.60, 1.8, 0.4]])  # eps 0.47 and 0.80
        args = (np.full(2, 85.0), np.full(2, 600.0), np.full(2, 2e4), "MAST")
        free, capped = raw_scalars(shape, *args), raw_scalars(shape, *args, eps_max=0.7)
        assert np.allclose(capped[0], free[0]) and np.allclose(capped[:, 1], free[:, 1])
        e = np.array([0.7, 0.8])
        toroidal = (1.17 - 0.65 * e) / (1 - e**2) ** 2
        assert capped[1, 0] / free[1, 0] == pytest.approx(toroidal[0] / toroidal[1])

    def test_volume_average_of_analytic_profiles(self):
        rho = np.linspace(0.0, 0.98, 50)
        assert volume_average(rho, 1 - rho**2) == pytest.approx(0.5, abs=5e-3)  # parabolic: <p> = p0 / 2
        assert volume_average(np.r_[rho, 1.2, 0.5], np.r_[1 - rho**2, 9.0, np.nan]) == pytest.approx(0.5, abs=5e-3)
        assert np.isnan(volume_average(np.array([0.1, 0.2]), np.array([1.0, 1.0])))  # too few points


class TestMastGeometry:
    """FreeGS's MAST description, the reference the shipped coil geometry is audited against."""

    def test_solenoid_level_matches_long_solenoid_limit(self):
        # just outside a long thin solenoid at mid-height, psi -> mu0 * N * Rs^2 / (2 L) per ampere
        spec = SolenoidSpec(0.15, -5.0, 5.0, 2000)
        psi = spec.flux(np.array([0.5]), np.array([0.0]))[0, 0]
        assert psi == pytest.approx(4e-7 * np.pi * 2000 * 0.15**2 / (2 * 10.0) * 1e3, rel=0.01)

    def test_basis_matches_single_filament_and_skips_unknown_columns(self):
        cols = ["magnetics_p3u_current", "magnetics_tf_current"]
        basis = freegs_basis(GRID_R, GRID_Z, cols)
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        np.testing.assert_allclose(basis[0], filament_flux(RR, ZZ, 1.1, 1.1) * 1e3)
        assert not basis[1].any()

    def test_basis_diff_splits_level_from_shape(self):
        ref = np.linspace(1.0, 2.0, 25).reshape(5, 5)
        diff = BasisDiff.of(2 * ref, ref)
        assert diff.level_ratio == pytest.approx(2.0) and diff.level_diff == pytest.approx(ref.mean())
        assert diff.shape_rms == pytest.approx(ref.std())
        offset = BasisDiff.of(ref + 0.3, ref)
        assert offset.shape_rms == pytest.approx(0.0, abs=1e-12) and offset.level_diff == pytest.approx(0.3)


class TestVessel:
    RING = PassiveSet(*(np.array([v]) for v in (1.0, 0.0, 0.01, 0.1, 7e-7)))

    def test_ring_under_a_flux_ramp_settles_at_v_over_r_and_decays_with_l_over_r(self):
        tau = float(self.RING.inductance[0, 0] / self.RING.resistance[0])
        assert self.RING.time_constants() == pytest.approx([tau])
        dt, rate = tau / 200, 0.01  # Wb/rad per s
        n_ramp = 2000  # 10 tau
        psi = np.r_[rate * dt * np.arange(n_ramp), np.full(400, rate * dt * (n_ramp - 1))][:, None]
        cur = self.RING.induced(dt, psi)[:, 0]
        steady = -2 * np.pi * rate / self.RING.resistance[0]
        assert cur[n_ramp - 1] == pytest.approx(steady, rel=1e-3)
        assert cur[n_ramp - 1 + 200] == pytest.approx(steady / np.e, rel=0.01)  # one tau after the ramp stops

    def test_shell_and_plates_conduct_like_the_continuous_wall(self):
        rho, t = 7e-7, 0.01
        shell = cylinder_shell(2.0, -2.0, 2.0, t, rho, 16)
        plate = annular_plate(2.0, 0.2, 2.0, t, rho, 30)
        R_mid = 2.0 + t / 2
        assert (1 / shell.resistance).sum() == pytest.approx(4.0 * t / (rho * 2 * np.pi * R_mid))
        assert (1 / plate.resistance).sum() == pytest.approx(
            t * np.log(10) / (rho * 2 * np.pi), rel=5e-3
        )  # midpoint rule
        both = join(shell, plate)
        assert len(both.R) == 46 and np.all(np.linalg.eigvalsh(both.inductance) > 0)


class TestExtractionRegion:
    """Geometry-only extraction regions, the per-shot gap fill and the label-free tracking statistics."""

    def test_line_of_sight_hides_cells_behind_a_coil(self):
        blocked = coil_cells(MAST_R, MAST_Z, P2_COILS)
        seen = line_of_sight(blocked, MAST_R, MAST_Z, (0.5, 0.0))
        col = np.argmin(np.abs(MAST_R - 0.5))
        rows = [np.argmin(np.abs(MAST_Z - z)) for z in (1.75, 1.95, 1.0, -1.95)]
        assert blocked[rows[0], col] and not blocked[rows[2], col]
        assert seen[rows[2], col] and not seen[rows[1], col] and not seen[rows[3], col]

    def test_inside_polygon_and_the_diii_d_wall(self):
        square = inside_polygon(GRID_R, GRID_Z, np.array([1.0, 2.0, 2.0, 1.0]), np.array([-1.01, -1.01, 1.01, 1.01]))
        RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
        np.testing.assert_array_equal(square, (RR > 1.0) & (RR < 2.0) & (np.abs(ZZ) < 1.01))
        wall = Region.diii_d_wall(GRID_R, GRID_Z).mask
        cell = {
            p: wall[np.argmin(np.abs(GRID_Z - p[1])), np.argmin(np.abs(GRID_R - p[0]))]
            for p in [(1.7, 0.0), (2.45, 0.0), (1.2, 1.5), (0.9, 0.0)]
        }
        assert cell == {(1.7, 0.0): True, (2.45, 0.0): False, (1.2, 1.5): False, (0.9, 0.0): False}

    def test_mast_chamber_keeps_the_plasma_off_a_coil(self):
        psi = mast_psi_with_coil_peak()[None]
        free = lcfs_shapes(psi, MAST_R, MAST_Z, "MAST", MAST_AXIS_SIGN)[0]
        mask = Region.mast("chamber", MAST_R, MAST_Z, P2_COILS).mask
        chamber = lcfs_shapes(psi, MAST_R, MAST_Z, "MAST", MAST_AXIS_SIGN, mask, mask.astype(float))[0]
        assert free[0] == pytest.approx(0.5, abs=0.1)  # the unbounded extractor closes around the coil
        assert chamber[0] == pytest.approx(0.9, abs=0.1) and chamber[1] > 0.3

    def test_unknown_or_coil_less_chamber_region_raises(self):
        with pytest.raises(ValueError):
            Region.mast("chamber", MAST_R, MAST_Z, None)
        with pytest.raises(ValueError):
            Region.mast("envelope", MAST_R, MAST_Z, P2_COILS)

    def test_fill_from_shot_interpolates_within_a_shot_and_falls_back_to_thomson(self):
        shape = np.array([[np.nan] * 4, [0.9, 0.5, 1.8, 0.4], [np.nan] * 4, [1.0, 0.6, 2.0, 0.2], [np.nan] * 4])
        shot_index = np.array([0, 0, 0, 0, 1])
        thomson = np.array([[0.0, 0.0]] * 4 + [[0.85, 0.55]])
        filled, source = fill_from_shot(shape, shot_index, thomson)
        np.testing.assert_allclose(filled[0], shape[1])  # held at the shot's start
        np.testing.assert_allclose(filled[2], (shape[1] + shape[3]) / 2)
        np.testing.assert_allclose(filled[4], [0.85, 0.55, 1.9, 0.3])  # Thomson size, median plausible shape
        assert source.tolist() == [1, 0, 1, 0, 2]

    def test_tracking_stats_separate_smooth_from_noisy_and_ignore_shot_changes(self):
        n = 400
        shot_index = np.repeat([0, 1], n // 2)
        ramp = np.c_[np.r_[np.linspace(3, 5, n // 2), np.linspace(8, 10, n // 2)], np.full(n, 1.0)]
        ramp[:, 1] += np.random.default_rng(0).normal(size=n)
        stats = tracking_stats(ramp, ramp, shot_index)
        assert stats["q95_vn"] < 0.01 and stats["q95_step"] == pytest.approx(2 / (n // 2 - 1), rel=1e-3)
        assert stats["betaN_vn"] == pytest.approx(1.0, abs=0.2)
        assert stats["q95_corr"] == pytest.approx(1.0) and stats["q95_mad"] == 0.0


class TestRule3:
    """No MAST prediction path may read the scorer's MAST envelope."""

    @pytest.fixture(autouse=True)
    def no_mast_envelope(self, monkeypatch):
        load = np.load

        def guarded(path, *args, **kwargs):
            if "mast_envelope" in str(path):
                raise AssertionError(f"MAST prediction path read {path}")
            return load(path, *args, **kwargs)

        monkeypatch.setattr(np, "load", guarded)

    def test_mast_boundary_shapes_extract_without_the_envelope(self, tmp_path):
        R, Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = MAST_AXIS_SIGN * -(((RR - 0.9) / 0.5) ** 2 + (ZZ / 0.9) ** 2)  # axis a maximum of MAST_AXIS_SIGN * psi
        (tmp_path / "mast_public_test").mkdir()
        for name, arr in (("grid_R", R), ("grid_Z", Z), ("psi_x", np.stack([psi, psi]).astype(np.float16))):
            np.save(tmp_path / "mast_public_test" / f"{name}.npy", arr)
        shapes = MastCache(tmp_path).boundary_shapes("psi_x.npy", workers=1)
        assert np.isfinite(shapes).all() and shapes[:, 2] == pytest.approx([1.8, 1.8], abs=0.1)

    def test_mast_chamber_boundary_shapes_build_without_the_envelope(self, tmp_path):
        (tmp_path / "mast_public_test").mkdir()
        psi = mast_psi_with_coil_peak()
        for name, arr in (("grid_R", MAST_R), ("grid_Z", MAST_Z), ("psi_x", np.stack([psi, psi]).astype(np.float16))):
            np.save(tmp_path / "mast_public_test" / f"{name}.npy", arr)
        cache = MastCache(tmp_path)
        cache.coils = P2_COILS
        shapes = cache.boundary_shapes("psi_x.npy", workers=1, region="chamber")
        assert shapes[:, 0] == pytest.approx([0.9, 0.9], abs=0.1)
        assert (tmp_path / "mast_public_test" / "lcfs_psi_x_chamber.npy").exists()

    def test_plausible_boundary_drops_coil_and_null_contours(self):
        thomson = np.array([[0.9, 0.5]] * 4)
        shape = np.array([[0.9, 0.5, 1.8, 0.4], [0.5, 0.05, 1.5, 0.0], [0.95, 1.0, 1.9, 0.3], [0.9, 0.55, 3.4, 0.4]])
        kept = ~np.isnan(plausible_boundary(shape, thomson)).any(1)
        assert kept.tolist() == [True, False, False, False]

    def test_agnostic_mast_frame_builds_without_the_envelope(self):
        assert MachineFrame.for_machine("MAST").envelope.any()
