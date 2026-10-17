"""Ensembles and blends of saved DIII-D models, and the per-config .npz a Codabench submission points at."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
import typer
from loguru import logger

from fusioneq.blend import Blend, BlendSpec, CoefTrees
from fusioneq.cache import Frames
from fusioneq.evaluation import HeldOut
from fusioneq.loading import load_model
from fusioneq.model import Ensemble, NeuralPredictor, Predictor, Smoothed
from fusioneq.physics import MastCache
from fusioneq.settings import CACHE_ROOT, RESULTS, SUBMISSION_DIR
from fusioneq.tracking import Run

Prediction = tuple[np.ndarray, np.ndarray]


def combine(preds: list[Prediction], agg: Literal["mean", "median"]) -> Prediction:
    """Mean or median of members' (psi, scalars) predictions over the member axis."""
    reduce = np.mean if agg == "mean" else np.median
    return tuple(reduce(np.stack([p[k] for p in preds]), axis=0).astype(np.float32) for k in (0, 1))


def record_model(ev: HeldOut, model: Predictor, tag: str, record: dict[str, Any], t0: float, job_type: str) -> None:
    ev.record(tag, model, record, Run(tag, {**record, "holdout": ev.holdout}, job_type=job_type), t0)


def record_ensemble(ev: HeldOut, model: Ensemble, tag: str, record: dict[str, Any], t0: float) -> None:
    record = {"model": "ensemble", "members": [p.name for p in model.paths], **record}
    record_model(ev, model, tag, record, t0, "ensemble")


def ensemble(
    members: Annotated[str, typer.Option(help="comma-separated result tags to average")],
    tag: str = "ensemble",
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Average saved models' psi and scalars, score on the same held-out shots, save as results/<tag>."""
    t0 = time.time()
    record_ensemble(
        HeldOut.load(cache_root, holdout, score_shots), Ensemble([RESULTS / m for m in members.split(",")]), tag, {}, t0
    )


def ensemble_search(
    members: Annotated[str, typer.Option(help="comma-separated result tags of candidate members")],
    tag: str = "ensemble_search",
    top_ks: str = "3,5,7",
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Choose which members to average (all, or the best k) on held-out shots that are never scored, so the choice
    cannot overfit the scoring shots; then score the winner on those and save it. The median is reported only."""
    t0, ev = time.time(), HeldOut.load(cache_root, holdout, score_shots)
    val = ev.validation
    paths = [RESULTS / m for m in members.split(",")]
    preds = [NeuralPredictor.load(p).predict(ev.fr, val) for p in paths]
    solo = np.array([ev.score(*p, mask=val)["S"] for p in preds])
    order = [int(i) for i in np.argsort(solo)[::-1]]
    logger.info("validation S per member: " + ", ".join(f"{paths[i].name} {solo[i]:.5f}" for i in order))
    everyone = list(range(len(paths)))
    choices = {"mean": (everyone, "mean"), "median": (everyone, "median")}
    choices |= {f"top{k}": (order[:k], "mean") for k in map(int, top_ks.split(",")) if k < len(paths)}
    val_s = {n: ev.score(*combine([preds[i] for i in idx], agg), mask=val)["S"] for n, (idx, agg) in choices.items()}
    logger.info("validation S per choice: " + ", ".join(f"{n} {v:.5f}" for n, v in val_s.items()))
    best = max((n for n, (_, agg) in choices.items() if agg == "mean"), key=val_s.get)
    model = Ensemble([paths[i] for i in choices[best][0]])
    record_ensemble(ev, model, tag, {"choice": best, "validation_S": val_s}, t0)


def fit_trees(
    host: Annotated[str, typer.Option(help="result tag of the network whose inputs and coefficients the trees use")],
    out: Annotated[str, typer.Option(help="file name under results/, e.g. xgb_top32.json")],
    k: int = 32,
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """XGBoost on `host`'s leading k coefficients: every 4th training frame, early stopping on the never-scored
    validation shots."""
    ev = HeldOut.load(cache_root, holdout, score_shots)
    train, val = np.nonzero(ev.train)[0][::4], np.nonzero(ev.validation)[0]
    CoefTrees.fit(NeuralPredictor.load(RESULTS / host), ev.fr, train, val, RESULTS / out, k)


def compact_trees(names: Annotated[list[str], typer.Argument(help="JSON forests under results/")]) -> None:
    """Binary (.ubj) copies of saved XGBoost forests, which blends then load at a fifth of the memory."""
    for name in names:
        logger.info(f"{name} -> {CoefTrees.compact(RESULTS / name)}")


def blend(
    spec: Annotated[Path, typer.Option(help="blend yaml, e.g. configs/blends/7a56d52a.yaml")],
    tag: str = "blend",
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    score: bool = True,
) -> None:
    """Score a blend (see fusioneq.blend) on the never-scored validation shots and the scoring shots; save as
    results/<tag>, which `build --model` reads. --no-score only saves the spec (a blend chosen on the cached held-out
    components needs no re-scoring, and skipping it avoids loading the training frames)."""
    if not score:
        BlendSpec.from_yaml(spec).save(RESULTS / tag)
        logger.info(f"{tag}: spec saved without scoring")
        return
    t0, ev = time.time(), HeldOut.load(cache_root, holdout, score_shots)
    model = Blend.from_spec(BlendSpec.from_yaml(spec))
    val = ev.score(*model.predict(ev.fr, ev.validation), mask=ev.validation)["S"]
    logger.info(f"{tag}: validation S {val:.6f}")
    record_model(ev, model, tag, {"model": "blend", "spec": str(spec), "validation_S": val}, t0, "blend")


@dataclass
class Submission:
    """The per-config .npz a pointer submission serves: a model's DIII-D predictions, and for MAST (which the DIII-D
    model cannot ingest) a flux and scalar file already built in the MAST cache."""

    model: Path
    cache_root: Path = CACHE_ROOT
    out_dir: Path = SUBMISSION_DIR
    mast_psi: str = "psi_phys.npy"
    mast_scal: str = "scal_phys.npy"
    smooth_sigma: float = 0.0
    psi_dtype: str = "float16"

    def prediction(self, config: str, fr: Frames) -> tuple[Prediction, str]:
        if config.startswith("mast"):
            cache = MastCache(self.cache_root, config=config)
            return cache.prediction(self.mast_psi, self.mast_scal), f"{self.mast_psi}, {self.mast_scal}"
        model = Smoothed(load_model(self.model), self.smooth_sigma)
        psi_scal = self.predict_by_shots(model, fr, self.psi_dtype)
        return psi_scal, f"model {self.model.name}, sigma {self.smooth_sigma}, psi {self.psi_dtype}"

    @staticmethod
    def predict_by_shots(
        model: Predictor, fr: Frames, psi_dtype: npt.DTypeLike = np.float16, shots_per_chunk: int = 20
    ) -> Prediction:
        """The model's prediction for every frame (psi in `psi_dtype`), made a few whole shots at a time so a large
        blend's intermediate arrays fit on a shared host; whole shots keep context and smoothing intact. Logs each
        chunk, so an hour-long CPU build is never silent."""
        psi = np.empty((len(fr.shot_index), *fr.psi_vac.shape[1:]), psi_dtype)
        scal = np.empty((len(fr.shot_index), 2), np.float32)
        chunks = np.array_split(np.arange(fr.n_shots), max(1, -(-fr.n_shots // shots_per_chunk)))
        start = time.time()
        for i, chunk in enumerate(chunks, 1):
            rows = np.isin(fr.shot_index, chunk)
            psi[rows], scal[rows] = model.predict(fr, rows)
            elapsed = time.time() - start
            eta = elapsed * (len(chunks) - i) / i
            logger.info(f"predict: chunk {i}/{len(chunks)} ({elapsed / 60:.0f} min, ~{eta / 60:.0f} min left)")
        return psi, scal

    def build(self, configs: list[str]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for config in configs:
            fr = Frames.load(self.cache_root / config)
            (psi, scal), how = self.prediction(config, fr)
            out = self.out_dir / f"{config}.npz"
            self.write_npz(fr, psi, scal, out)
            logger.info(f"{config}: {fr.n_shots} shots, {len(psi)} frames ({how}) -> {out}")

    @staticmethod
    def write_npz(fr: Frames, psi: np.ndarray, scal: np.ndarray, out: Path) -> None:
        """Per-shot keys in HF stream order; psi in its own float dtype (never rounded), scalars float32."""
        preds = {}
        for i in range(fr.n_shots):
            rows = fr.shot_index == i
            preds[f"shot_{i:04d}_psirz"] = psi[rows]
            preds[f"shot_{i:04d}_q95"] = scal[rows, 0].astype(np.float32)
            preds[f"shot_{i:04d}_betaN"] = scal[rows, 1].astype(np.float32)
        np.savez_compressed(out, **preds)


def build(
    model: Path | None = None,
    cache_root: Path = CACHE_ROOT,
    out_dir: Path = SUBMISSION_DIR,
    configs: str = "diii_d_public_test,mast_public_test",
    mast_psi: str = "psi_phys.npy",
    mast_scal: str = "scal_phys.npy",
    smooth_sigma: float = 0.0,
    psi_dtype: str = "float16",
) -> None:
    """Per-config .npz: the model on DIII-D, smoothed over time within each shot by `smooth_sigma` frames; MAST gets
    `mast_psi` and `mast_scal` from its cache (the physics flux by default, or an agnostic network's psi_<tag>.npy,
    and a `mast-scalars` output). `psi_dtype` float32 keeps the DIII-D flux at full precision (float16 is ~2.5 GB
    smaller but rounds the magnetic axis)."""
    model = model or RESULTS / "pca_ridge.joblib"
    Submission(model, cache_root, out_dir, mast_psi, mast_scal, smooth_sigma, psi_dtype).build(configs.split(","))


COMMANDS = [ensemble, ensemble_search, fit_trees, compact_trees, blend, build]
