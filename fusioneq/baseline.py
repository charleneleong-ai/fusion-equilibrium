"""PCA + ridge on the plasma residual psi - psi_vac, plus ridge heads for q95 and betaN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fusioneq.cache import Frames
from fusioneq.targets import fit_pca, reconstruct, transform_all


@dataclass
class PcaRidge:
    n_components: int = 64
    alpha: float = 1.0
    use_vacuum: bool = True

    def fit(self, fr: Frames, mask: np.ndarray) -> PcaRidge:
        rows = np.nonzero(mask)[0]
        X = fr.features(rows)
        self.scaler = StandardScaler().fit(X)
        self.pca = fit_pca(fr, rows, self.n_components, self.use_vacuum)
        coef = transform_all(self.pca, fr, rows, self.use_vacuum)
        Xs = self.scaler.transform(X)
        self.ridge = Ridge(self.alpha).fit(Xs, coef)
        s = fr.scalars[rows][:, :2]
        ok = np.isfinite(s).all(1)
        self.heads = Ridge(self.alpha).fit(Xs[ok], s[ok])
        return self

    def predict(self, fr: Frames, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.nonzero(mask)[0]
        Xs = self.scaler.transform(fr.features(rows))
        coef = self.ridge.predict(Xs).astype(np.float32)
        return reconstruct(self.pca, coef, fr, rows, self.use_vacuum), self.heads.predict(Xs).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> PcaRidge:
        return joblib.load(path)
