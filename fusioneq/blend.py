"""The submitted DIII-D blend: the network ensemble, XGBoost trees on a network's leading PCA coefficients (a second
model family, so its errors decorrelate), and the shape-target hybrid - a shape-target ensemble's flux moved onto the
base flux's envelope mean and spread, so it adds shape (consistency, boundary) without its weaker level and scale.

    base = (1 - trees.weight) * ensemble + trees.weight * trees
    psi  = (1 - hybrid_weight - shape_trees.weight) * base + hybrid_weight * hybrid + shape_trees.weight * shape trees

with optional grid shape models (fusioneq.grid, averaged) taking grid.weight out of the hybrid's share.

Scalars come from the base ensemble. Saved as results/<tag>/blend.yaml (member tags, resolved against RESULTS)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
import yaml

from fusioneq.cache import Frames
from fusioneq.grid import GridEnsemble
from fusioneq.model import Ensemble, NeuralPredictor, Predictor
from fusioneq.settings import RESULTS
from fusioneq.targets import match_envelope, shape_stats

TREE_PARAMS: dict[str, Any] = dict(
    tree_method="hist",
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=4,
    early_stopping_rounds=100,
)


class CoefTrees:
    """XGBoost on a host network's inputs for its leading PCA coefficients, in units of the host's c_std (the PCA part
    of c_mean is zero); the host network supplies the rest of the block (and a shape-target host's stats)."""

    def __init__(self, host: NeuralPredictor, trees: xgb.XGBRegressor) -> None:
        self.host, self.trees = host, trees

    @classmethod
    def load(cls, host: NeuralPredictor, path: Path) -> CoefTrees:
        """Reads the compact (.ubj) copy beside `path` when there is one: a 2.3 GB JSON forest peaks at 28 GB to
        parse, its .ubj at 5.6 GB."""
        trees = xgb.XGBRegressor()
        trees.load_model(ubj if (ubj := path.with_suffix(".ubj")).exists() else path)
        trees.set_params(device=host.device)
        return cls(host, trees)

    @staticmethod
    def compact(path: Path) -> Path:
        """Write the binary (.ubj) copy of a JSON forest that `load` prefers; predictions are unchanged."""
        trees = xgb.Booster()
        trees.load_model(path)
        trees.save_model(out := path.with_suffix(".ubj"))
        return out

    @staticmethod
    def fit(host: NeuralPredictor, fr: Frames, train: np.ndarray, val: np.ndarray, out: Path, k: int = 32) -> None:
        """Fit on `train` rows with early stopping on `val` rows; save to `out`."""

        def data(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return host.inputs(fr, rows), host.coef_block(fr, rows)[:, :k] / host.c_std[:k]

        trees = xgb.XGBRegressor(device=host.device, **TREE_PARAMS)
        trees.fit(*data(train), eval_set=[data(val)], verbose=500)
        trees.save_model(out)

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.nonzero(mask)[0]
        X = self.host.inputs(fr, rows)
        block, scal = self.host.coefficients(fr, mask, X)
        lead = self.trees.predict(X)
        k = lead.shape[1]
        block[:, :k] = lead * self.host.c_std[:k]
        return self.host.decode(block, fr, rows), scal


@dataclass(frozen=True)
class TreeSpec:
    host: str
    model: str
    weight: float


@dataclass(frozen=True)
class GridSpec:
    models: list[str]  # grid models (results tags), averaged
    weight: float  # taken out of hybrid_weight


@dataclass(frozen=True)
class BlendSpec:
    base: list[str]
    shape: list[str]
    hybrid_weight: float
    trees: TreeSpec | None = None
    shape_trees: TreeSpec | None = None
    grid: GridSpec | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> BlendSpec:
        raw = yaml.safe_load(path.read_text())
        parts = {"trees": TreeSpec, "shape_trees": TreeSpec, "grid": GridSpec}
        return cls(**{**raw, **{k: part(**raw[k]) for k, part in parts.items() if raw.get(k)}})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "blend.yaml").write_text(yaml.safe_dump(asdict(self), sort_keys=False))


class Blend:
    """The weights live in `spec`; `trees` / `shape_trees` / `grid` must be given when the spec names them."""

    def __init__(
        self,
        spec: BlendSpec,
        base: Predictor,
        shape: Predictor,
        trees: Predictor | None = None,
        shape_trees: Predictor | None = None,
        grid: GridEnsemble | None = None,
    ) -> None:
        self.spec, self.base, self.shape = spec, base, shape
        self.trees, self.shape_trees, self.grid = trees, shape_trees, grid

    @classmethod
    def from_spec(cls, spec: BlendSpec) -> Blend:
        base, shape = (Ensemble([RESULTS / t for t in tags]) for tags in (spec.base, spec.shape))
        loaded = {p.name: m for e in (base, shape) for p, m in zip(e.paths, e.members, strict=True)}

        def trees(t: TreeSpec | None) -> CoefTrees | None:
            if t is None:
                return None
            host = loaded.get(t.host) or NeuralPredictor.load(RESULTS / t.host)
            return CoefTrees.load(host, RESULTS / t.model)

        grid = GridEnsemble.load([RESULTS / m for m in spec.grid.models]) if spec.grid else None
        return cls(spec, base, shape, trees(spec.trees), trees(spec.shape_trees), grid)

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.spec
        psi, scal = self.base.predict(fr, mask)
        if s.trees:
            psi *= 1 - s.trees.weight
            psi += s.trees.weight * self.trees.predict(fr, mask)[0]
        stats = shape_stats(psi.reshape(len(psi), -1))
        w_st = s.shape_trees.weight if s.shape_trees else 0.0
        w_g = s.grid.weight if s.grid else 0.0
        psi *= 1 - s.hybrid_weight - w_st
        for model, w in ((self.shape, s.hybrid_weight - w_g), (self.shape_trees, w_st), (self.grid, w_g)):
            if w:
                part = match_envelope(self.shape_of(model, fr, mask), stats)
                part *= w
                psi += part
        return psi, scal

    @staticmethod
    def shape_of(model: Predictor | GridEnsemble, fr: Frames, mask: np.ndarray) -> np.ndarray:
        """A shape component's flux: shape-only models (predict_shape) as they are, full predictors' psi otherwise."""
        return model.predict_shape(fr, mask) if hasattr(model, "predict_shape") else model.predict(fr, mask)[0]

    def save(self, path: Path) -> None:
        self.spec.save(path)

    @classmethod
    def load(cls, path: Path) -> Blend:
        return cls.from_spec(BlendSpec.from_yaml(path / "blend.yaml"))
