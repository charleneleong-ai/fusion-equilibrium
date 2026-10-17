"""Load any saved model by what its results entry holds: blend.yaml, members.json, a network directory or a .joblib."""

from __future__ import annotations

from pathlib import Path

from fusioneq.baseline import PcaRidge
from fusioneq.blend import Blend
from fusioneq.model import Ensemble, NeuralPredictor


def load_predictor(path: Path) -> NeuralPredictor | Ensemble | Blend:
    if (path / "blend.yaml").exists():
        return Blend.load(path)
    return Ensemble.load(path) if (path / "members.json").exists() else NeuralPredictor.load(path)


def load_model(path: Path) -> NeuralPredictor | Ensemble | Blend | PcaRidge:
    """A saved model of any kind: a results directory (network, ensemble or blend) or a PCA-ridge .joblib."""
    return load_predictor(path) if path.is_dir() else PcaRidge.load(path)
