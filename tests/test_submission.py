"""Ensembles, the DIII-D blend and the submission .npz."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

from fusioneq.blend import Blend, BlendSpec, CoefTrees, GridSpec, TreeSpec
from fusioneq.budget import LIMIT_HOURS, RebuildBudget
from fusioneq.commands.submission import Submission, combine
from fusioneq.grid import GridEnsemble
from fusioneq.model import smooth_in_time
from fusioneq.targets import match_envelope, shape_stats

BLENDS = Path(__file__).parents[1] / "configs" / "blends"


class Fixed:
    """A model stand-in that predicts a fresh copy of a fixed psi (blends work in place) and zero scalars."""

    def __init__(self, psi: np.ndarray) -> None:
        self.psi = psi

    def predict(self, fr: object, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.psi.copy(), np.zeros((len(self.psi), 2), np.float32)


class FixedShape:
    """A shape-only model stand-in (like the grid model): predict_shape returns a fresh copy of a fixed map."""

    def __init__(self, psi: np.ndarray) -> None:
        self.psi = psi

    def predict_shape(self, fr: object, mask: np.ndarray) -> np.ndarray:
        return self.psi.copy()


class RowsSeen:
    """A model stand-in whose psi and scalars encode each frame's row and shot, so a misplaced chunk shows up."""

    def predict(self, fr: SimpleNamespace, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.nonzero(mask)[0]
        psi = rows[:, None, None] * np.ones((1, 2, 2), np.float32)
        return psi, np.stack([rows, fr.shot_index[mask]], 1).astype(np.float32)


def envelope_stats(psi: np.ndarray) -> np.ndarray:
    return shape_stats(psi.reshape(len(psi), -1))


@pytest.fixture
def flux() -> tuple[np.ndarray, np.ndarray]:
    """Two unrelated (4, 65, 65) flux stacks with different levels and spreads."""
    rng = np.random.default_rng(0)
    return rng.normal(1.0, 0.3, (4, 65, 65)).astype(np.float32), rng.normal(-2.0, 2.0, (4, 65, 65)).astype(np.float32)


class TestEnsemble:
    def test_median_ignores_one_outlier_member_the_mean_does_not(self):
        psi = [np.full((2, 3, 3), v, np.float32) for v in (1.0, 1.2, 50.0)]
        preds = [(p, p[:, 0, :2]) for p in psi]
        assert np.allclose(combine(preds, "median")[0], 1.2) and np.allclose(combine(preds, "mean")[1], 52.2 / 3)

    def test_npz_has_per_shot_psi_q95_betan_keys_in_shot_order(self, tmp_path):
        fr = SimpleNamespace(n_shots=2, shot_index=np.array([0, 0, 1]))
        psi = np.arange(3 * 4, dtype=np.float16).reshape(3, 2, 2)
        scal = np.array([[4.0, 1.0], [5.0, 2.0], [6.0, 3.0]])
        Submission.write_npz(fr, psi, scal, tmp_path / "x.npz")
        z = np.load(tmp_path / "x.npz")
        assert z.files == [f"shot_{i:04d}_{k}" for i in range(2) for k in ("psirz", "q95", "betaN")]
        assert z["shot_0000_psirz"].dtype == np.float16 and z["shot_0001_psirz"].shape == (1, 2, 2)
        assert z["shot_0001_q95"].tolist() == [6.0] and z["shot_0000_betaN"].dtype == np.float32

    def test_prediction_by_shot_chunks_matches_one_pass(self):
        shots = np.repeat(np.arange(5), [3, 1, 4, 2, 3])
        fr = SimpleNamespace(n_shots=5, shot_index=shots, psi_vac=np.zeros((len(shots), 2, 2)))
        psi, scal = Submission.predict_by_shots(RowsSeen(), fr, shots_per_chunk=2)
        whole_psi, whole_scal = RowsSeen().predict(fr, np.ones(len(shots), bool))
        assert psi.dtype == np.float16 and np.array_equal(psi, whole_psi) and np.array_equal(scal, whole_scal)

    def test_float32_psi_is_written_unrounded(self, tmp_path):
        fr = SimpleNamespace(n_shots=1, shot_index=np.zeros(3, int), psi_vac=np.zeros((3, 2, 2)))
        psi, scal = Submission.predict_by_shots(RowsSeen(), fr, np.float32)
        psi += 1e-4  # below float16 resolution at this scale
        Submission.write_npz(fr, psi, scal, tmp_path / "x.npz")
        z = np.load(tmp_path / "x.npz")["shot_0000_psirz"]
        assert z.dtype == np.float32 and np.array_equal(z, psi)


class TestSmoothing:
    def test_smooths_noise_within_a_shot_without_leaking_across_shots(self):
        rng = np.random.default_rng(0)
        shot_index = np.repeat([0, 1], 50)
        x = np.where(shot_index == 0, 0.0, 10.0)[:, None] + rng.normal(0, 1, (100, 1))
        y = smooth_in_time(x, shot_index, 2.0)
        assert y[shot_index == 0].std() < 0.7 * x[shot_index == 0].std()
        assert abs(y[49, 0] - x[:50, 0].mean()) < 2 and abs(y[50, 0] - 10) < 2  # the step between shots survives
        assert smooth_in_time(x, shot_index, 0.0) is x


class TestBlend:
    def test_match_envelope_takes_the_level_and_spread_and_keeps_the_shape(self, flux):
        psi, like = flux
        out = match_envelope(psi, envelope_stats(like))
        assert np.allclose(envelope_stats(out), envelope_stats(like), atol=1e-4)
        assert np.allclose(match_envelope(out, envelope_stats(psi)), psi, atol=1e-4)  # only level and spread moved

    def test_affine_copies_of_the_base_leave_it_unchanged(self, flux):
        base, _ = flux
        spec = BlendSpec([], [], 0.65, shape_trees=TreeSpec("host", "trees.json", 0.1), grid=GridSpec(["a", "b"], 0.1))
        grid = GridEnsemble([FixedShape(2 * base + 5), FixedShape(4 * base - 1)])  # mean is again affine in base
        blend = Blend(spec, Fixed(base), Fixed(3 * base + 2), shape_trees=Fixed(0.5 * base - 1), grid=grid)
        assert np.allclose(blend.predict(None, None)[0], base, atol=1e-4)

    def test_trees_mix_linearly_into_the_base(self, flux):
        base, other = flux
        spec = BlendSpec([], [], hybrid_weight=0.0, trees=TreeSpec("host", "trees.json", 0.25))
        psi, _ = Blend(spec, Fixed(base), Fixed(base), trees=Fixed(other)).predict(None, None)
        assert np.allclose(psi, 0.75 * base + 0.25 * other, atol=1e-5)

    def test_trees_load_their_compact_copy_with_the_same_predictions(self, tmp_path):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(200, 5)), rng.normal(size=(200, 2))
        path = tmp_path / "trees.json"
        xgb.XGBRegressor(n_estimators=20, max_depth=3).fit(X, y).save_model(path)
        before = CoefTrees.load(SimpleNamespace(device="cpu"), path).trees.predict(X)
        CoefTrees.compact(path)
        path.write_text("{}")  # a load that still read the JSON would now fail
        assert np.array_equal(CoefTrees.load(SimpleNamespace(device="cpu"), path).trees.predict(X), before)

    @pytest.mark.parametrize("name", sorted(p.name for p in BLENDS.glob("*.yaml")))
    def test_committed_specs_round_trip_through_a_saved_spec(self, name, tmp_path):
        spec = BlendSpec.from_yaml(BLENDS / name)
        spec.save(tmp_path)
        assert BlendSpec.from_yaml(tmp_path / "blend.yaml") == spec
        assert not any(isinstance(v, dict) for v in vars(spec).values())  # every sub-spec parsed, not left raw


class TestRebuildBudget:
    """Training-hour ranges of a blend spec for the Final's 24 A100-hour cap."""

    def test_submitted_blend_counts_and_range(self) -> None:
        b = RebuildBudget.of(BlendSpec.from_yaml(BLENDS / "v21_27_grids3.yaml"))
        assert (b.mlps, b.forests, b.grids) == (42, 2, ("grid_v2", "grid_v2_s1", "grid_v3"))
        lo, hi = b.hours()
        assert 20 < lo < LIMIT_HOURS < hi

    def test_shared_member_counted_once_and_unknown_grid_rejected(self) -> None:
        spec = BlendSpec(base=["a", "b"], shape=["b"], hybrid_weight=0.75, grid=GridSpec(["grid_new"], 0.65))
        b = RebuildBudget.of(spec)
        assert (b.mlps, b.forests) == (2, 0)
        with pytest.raises(KeyError, match="grid_new"):
            b.hours()
