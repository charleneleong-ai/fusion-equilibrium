"""The DIII-D finishing step: the scorer's reading of a map, the displacement warp, the li remap and the gate."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from fusioneq.cache import Frames
from fusioneq.finish import CONS_INDEX, SCORER, Finisher, FinishSpec, Warp, polygon_shape, read_frame
from fusioneq.scalar_heads import HeadConfig, ScalarHeads
from fusioneq.score import symmetric_hausdorff

SCALE = np.array([0.04, 0.04, 0.1, 0.24, 0.22, 1.3, 0.25])
LI, R_AXIS, Z_AXIS = CONS_INDEX["li"], CONS_INDEX["R_axis"], CONS_INDEX["Z_axis"]


@pytest.fixture(scope="module")
def psi() -> np.ndarray:
    """A shaped plasma on the scorer's DIII-D grid, axis at the psi minimum (DIII-D's sign), in a tilted field."""
    RR, ZZ = np.meshgrid(SCORER.R, SCORER.Z)
    x, y = (RR - 1.7) / 0.6, ZZ / 1.08
    rho2 = (x + 0.2 * y**2) ** 2 + y**2
    return -(1 - rho2) * (1 + 0.2 * (1 - rho2)) * 2.0 + 0.3 * x


@pytest.fixture(scope="module")
def reading(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return read_frame(psi)


def test_reading_is_finite_with_a_closed_boundary(reading):
    v, c = reading
    assert np.isfinite(v).all() and c is not None and len(c) == 512


class TestWarp:
    def test_zero_displacement_returns_the_map_unchanged(self, psi, reading):
        v, c = reading
        warp = Warp.of((v[R_AXIS], v[Z_AXIS]), c)
        np.testing.assert_array_equal(warp.apply(psi, np.zeros(6), SCORER.R, SCORER.Z), psi)

    def test_axis_bumps_move_the_scorer_axis(self, psi, reading):
        v, c = reading
        warp = Warp.of((v[R_AXIS], v[Z_AXIS]), c)
        moved, _ = read_frame(warp.apply(psi, np.r_[0.006, -0.01, 0, 0, 0, 0], SCORER.R, SCORER.Z))
        assert moved[R_AXIS] - v[R_AXIS] == pytest.approx(0.006, abs=1.5e-3)
        assert moved[Z_AXIS] - v[Z_AXIS] == pytest.approx(-0.01, abs=2e-3)

    def test_solve_puts_the_advected_polygon_on_the_shape_targets(self, reading):
        v, c = reading
        warp = Warp.of((v[R_AXIS], v[Z_AXIS]), c)
        target = np.r_[0.0, 0.0, v[2] + 0.02, v[3] - 0.01, v[4] + 0.01, v[5] + 0.2]
        np.testing.assert_allclose(polygon_shape(warp.moved(c, warp.solve(c, target))), target[2:], atol=1e-6)


class TestFinisher:
    def test_li_target_is_hit_with_the_boundary_kept(self, psi, reading):
        v, c = reading
        target = np.full(7, np.nan)
        target[LI] = v[LI] + 0.03
        out, got, ok = Finisher(SCALE, shape=False).frame(psi, target)
        assert ok and got[LI] == pytest.approx(target[LI], abs=1e-3)
        assert symmetric_hausdorff(read_frame(out)[1], c) < 1e-3

    def test_a_frame_already_on_target_is_returned_untouched(self, psi, reading):
        out, got, ok = Finisher(SCALE).frame(psi, reading[0])
        assert not ok
        np.testing.assert_array_equal(out, psi)
        np.testing.assert_array_equal(got, reading[0])

    def test_weight_blends_the_head_into_the_reading_and_zero_leaves_a_scalar(self, psi, reading):
        v, _ = reading
        head = v.copy()
        head[LI] += 0.04
        weight = np.zeros(7)
        weight[LI] = 0.5
        _, got, ok = Finisher(SCALE, shape=False, weight=weight).frame(psi, head)
        assert ok and got[LI] == pytest.approx(v[LI] + 0.02, abs=1e-3)
        out, _, ok = Finisher(SCALE, weight=np.zeros(7)).frame(psi, head)
        assert not ok and out is psi

    def test_frame_without_a_boundary_is_left_alone(self):
        flat = np.broadcast_to(SCORER.R[None], (65, 65)).astype(np.float64)
        out, _, ok = Finisher(SCALE).frame(flat, np.zeros(7))
        assert not ok and out is flat


def test_spec_weights_only_its_operators_and_names_one_head_per_seed(tmp_path):
    spec = FinishSpec({"epochs": 1}, 2, (0, 3), {"R_axis": 0.3, "kappa": 0.4, "li": 0.2}, ops=("axis", "li"))
    spec.save(tmp_path / "axis_li.json")
    loaded = FinishSpec.load(tmp_path / "axis_li.json")
    assert loaded == spec and loaded.head_tags() == ["heads_axis_li_s0", "heads_axis_li_s3"]
    np.testing.assert_array_equal(loaded.weight(), [0.3, 0, 0, 0, 0, 0, 0.2])
    fin = loaded.finisher(SCALE)
    assert fin.shape and fin.li and fin.jump == 2.0


def test_heads_learn_a_scalar_of_the_inputs_and_reload_identically(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    fr = Frames(
        currents=rng.normal(size=(n, 3)).astype(np.float32),
        ip=rng.normal(size=n).astype(np.float32),
        thomson=rng.normal(size=(n, 4)).astype(np.float32),
        psi_vac=None,
        shot_index=np.repeat(np.arange(6), 100),
        efit_times=np.tile(np.arange(100.0) * 20, 6),
        grid_R=None,
        grid_Z=None,
        shots=[],
    )
    y = np.stack([fr.currents[:, 0] + 0.5 * fr.ip] * 7, 1).astype(np.float64)
    y[::7, 3] = np.nan  # a failed derivation is masked, not learned
    rows = np.arange(n)
    cfg = HeadConfig(hidden=32, layers=2, epochs=60, batch=64, context=0, thomson_hr=False, history=False)
    heads = ScalarHeads(cfg, "cpu").fit(fr, rows, y)
    pred = heads.predict(fr, rows)
    assert np.corrcoef(pred[:, 0], y[:, 0])[0, 1] > 0.95 and np.isfinite(pred).all()
    heads.save(tmp_path / "h")
    np.testing.assert_array_equal(ScalarHeads.load(tmp_path / "h", "cpu").predict(fr, rows), pred)

    reading = y + rng.normal(0, 0.3, y.shape)  # a map reading off by noise the inputs explain nothing of
    reading[:, 0] = y[:, 0] + 0.5  # and by a constant bias the residual head should remove
    resid = ScalarHeads(HeadConfig(**{**asdict(cfg), "residual": True}), "cpu").fit(fr, rows, y, reading)
    out = resid.predict(fr, rows, reading)
    assert np.abs(out[:, 0] - y[:, 0]).mean() < 0.5 * np.abs(reading[:, 0] - y[:, 0]).mean()
    resid.save(tmp_path / "r")
    np.testing.assert_array_equal(ScalarHeads.load(tmp_path / "r", "cpu").predict(fr, rows, reading), out)
    with pytest.raises(ValueError, match="reading"):
        resid.predict(fr, rows)
