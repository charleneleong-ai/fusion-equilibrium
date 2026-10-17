"""The li lever: core flux remap to a target internal inductance with the LCFS kept."""

from __future__ import annotations

import numpy as np
import pytest

from fusioneq.li_shape import C_GRID, CoreShaper, points_in_polygon, shot_smooth, shrink_targets
from fusioneq.score import extract_lcfs_with_sign, internal_inductance, symmetric_hausdorff

GRID_R, GRID_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
GRID = np.asarray(C_GRID)
SIGN = -1.0  # MAST orientation: the axis is a minimum of psi


@pytest.fixture(scope="module")
def psi() -> np.ndarray:
    """A peaked plasma at (0.9, 0) in a field with an X-point below it, axis a minimum."""
    plasma = np.exp(-(((RR - 0.9) / 0.35) ** 2) - (ZZ / 0.6) ** 2)
    return SIGN * (plasma + 0.25 * ZZ * (ZZ < 0) * (RR > 0.3))


@pytest.fixture(scope="module")
def shaper() -> CoreShaper:
    return CoreShaper(GRID_R, GRID_Z, "MAST", SIGN)


@pytest.fixture(scope="module")
def remapped(psi: np.ndarray, shaper: CoreShaper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bump, contour = shaper.bump(psi)
    return bump, contour, shaper.curve(psi, bump, contour)


class TestCoreShaper:
    def test_curve_matches_the_scorer_li_and_rises_with_c(self, psi, shaper, remapped):
        bump, contour, curve = remapped
        for j in (0, 5, 20, len(GRID) - 1):
            want = internal_inductance(psi + GRID[j] * bump, GRID_R, GRID_Z, "MAST", contour)
            assert curve[j] == pytest.approx(want, rel=1e-9)
        assert np.all(np.diff(curve) > 0)

    @pytest.mark.parametrize("c", [-0.4, 1.0])
    def test_remap_keeps_the_lcfs_and_the_axis_cell(self, psi, remapped, c):
        bump, contour, _ = remapped
        new = psi + c * bump
        moved, _ = extract_lcfs_with_sign(new, GRID_R, GRID_Z, "MAST", axis_sign=SIGN)
        assert symmetric_hausdorff(moved, contour) < 0.02
        assert np.argmin(new) == np.argmin(psi)
        assert np.all(bump[~points_in_polygon(GRID_R, GRID_Z, contour)] == 0)

    def test_coefficient_hits_the_target_and_skips_missing(self, shaper, remapped):
        _, _, curve = remapped
        target = np.array([0.5 * (curve[3] + curve[4]), np.nan, curve[0] - 1.0])
        c = shaper.coefficient(np.stack([curve] * 3), target)
        assert np.interp(c[0], GRID, curve) == pytest.approx(target[0])
        assert c[1] == 0.0 and c[2] == GRID[0]

    def test_no_closed_surface_leaves_the_frame_alone(self, shaper):
        flat = np.broadcast_to(GRID_R[None], RR.shape).copy()
        bump, contour = shaper.bump(flat)
        assert contour is None and not bump.any()
        assert np.isnan(shaper.curve(flat, bump, contour)).all()


def test_shrink_targets_moves_the_level_and_scales_the_spread():
    li = np.array([1.0, 1.4, 1.8, np.nan])
    np.testing.assert_allclose(shrink_targets(li, 1.0, 0.5), [0.8, 1.0, 1.2, np.nan])


def test_shot_smooth_removes_a_spike_without_crossing_shots():
    v = np.array([1.0, 1.0, 5.0, 1.0, 1.0, 2.0, 2.0, np.nan])
    out = shot_smooth(v, np.array([0, 0, 0, 0, 0, 1, 1, 1]), 3)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(shot_smooth(v, np.zeros(8, int), 1), v)
