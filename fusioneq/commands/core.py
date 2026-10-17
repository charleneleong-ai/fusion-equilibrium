"""Core commands: build the frame caches, fit and score the PCA-ridge baseline and the neural predictor."""

from __future__ import annotations

import time
from pathlib import Path

from autoresearch.gpu_monitor import GPUMonitor
from loguru import logger

from fusioneq.baseline import PcaRidge
from fusioneq.evaluation import HeldOut, save_score
from fusioneq.loading import load_model
from fusioneq.model import NetConfig, NeuralPredictor
from fusioneq.prepare import prepare as prepare_config
from fusioneq.prepare import prepare_thomson_hr
from fusioneq.score import summary_line
from fusioneq.settings import CACHE_ROOT, DATA_ROOT, RESULTS, load_config
from fusioneq.targets import r2_psi
from fusioneq.timing import Phases, gpu_metrics
from fusioneq.tracking import Run
from fusioneq.viz import render_all


def prepare(
    config: str = "diii_d_train",
    data_root: Path = DATA_ROOT,
    cache_root: Path = CACHE_ROOT,
    limit: int | None = None,
    workers: int = 12,
) -> None:
    """Cache one config's frames (inputs, vacuum flux, labels) as memory-mapped arrays."""
    prepare_config(data_root, config, cache_root / config, limit, workers)


def prepare_thomson(
    config: str = "diii_d_train", data_root: Path = DATA_ROOT, cache_root: Path = CACHE_ROOT, window_ms: float = 20.0
) -> None:
    """Add full-resolution, time-averaged Thomson features (thomson_hr.npy) to an existing cache."""
    prepare_thomson_hr(data_root, config, cache_root / config, window_ms)


def baseline(
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    n_components: int = 64,
    alpha: float = 1.0,
    use_vacuum: bool = True,
    tag: str = "pca_ridge",
) -> None:
    """Fit the PCA + ridge baseline and score it on the held-out shots."""
    t0, ev = time.time(), HeldOut.load(cache_root, holdout, score_shots)
    record = {"model": "pca_ridge", "n_components": n_components, "alpha": alpha, "use_vacuum": use_vacuum}
    run = Run(tag, {**record, "holdout": holdout})
    model = PcaRidge(n_components, alpha, use_vacuum).fit(ev.fr, ev.train)
    RESULTS.mkdir(parents=True, exist_ok=True)
    model.save(RESULTS / f"{tag}.joblib")
    psi_pred, scal_pred = model.predict(ev.fr, ev.subset)
    r2 = r2_psi(ev.fr, ev.rows, psi_pred)
    logger.info(f"{tag}: frame-pooled R2psi on {score_shots} held-out shots = {r2:.4f}")
    res = ev.score(psi_pred, scal_pred)
    logger.info(f"{tag}: {summary_line(res)}")
    save_score(tag, record, res, run, t0, {"frame_r2_psi": float(r2)})
    run.finish()


def train(
    cache_root: Path = CACHE_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    tag: str = "mlp",
    n_components: int = 128,
    hidden: int = 1024,
    layers: int = 4,
    epochs: int = 20,
    batch: int = 2048,
    lr: float = 1e-3,
    context: int = 0,
    use_vacuum: bool = True,
    amp: bool = False,
    coef_loss: str = "std",
    region_weight: float = 0.0,
    seed: int = 0,
    thomson_hr: bool = False,
    history: bool = False,
    spare_holdout: bool = False,
    temporal_layers: int = 0,
    window: int = 64,
    ramp_weight: float = 1.0,
    normalize: bool = False,
    config: Path | None = None,
) -> None:
    """Fit the neural predictor on DIII-D train frames, score it on held-out shots, save it under results/<tag>.
    `--config` (a configs/members/ recipe) replaces every network flag."""
    t0, phases = time.time(), Phases()
    ev = HeldOut.load(cache_root, holdout, score_shots, spare_holdout)
    cfg = (
        load_config(NetConfig, config)
        if config
        else NetConfig(
            n_components=n_components,
            hidden=hidden,
            layers=layers,
            epochs=epochs,
            batch=batch,
            lr=lr,
            context=context,
            use_vacuum=use_vacuum,
            amp=amp,
            coef_loss=coef_loss,
            region_weight=region_weight,
            seed=seed,
            thomson_hr=thomson_hr,
            history=history,
            temporal_layers=temporal_layers,
            window=window,
            ramp_weight=ramp_weight,
            normalize=normalize,
        )
    )
    kind = "temporal" if temporal_layers else "mlp"
    run = Run(tag, {"model": kind, **cfg.__dict__, "holdout": holdout})
    with GPUMonitor() as gpu, phases("fit"):
        model = NeuralPredictor(cfg).fit(ev.fr, ev.train, ev.subset, run)
    RESULTS.mkdir(parents=True, exist_ok=True)
    model.save(RESULTS / tag)
    with phases("predict"):
        psi_pred, scal_pred = model.predict(ev.fr, ev.subset)
    with phases("score"):
        res = ev.score(psi_pred, scal_pred)
    gpu_stats, hints = gpu_metrics(gpu)
    timing = {**phases.as_metrics(), **gpu_stats}
    logger.info(f"{tag}: {summary_line(res)}")
    logger.info(f"{tag}: {timing}")
    for hint in hints:
        logger.info(f"{tag}: rightsizing: {hint}")
    record = {"model": kind, **cfg.__dict__, "spare_holdout": spare_holdout}
    save_score(tag, record, res, run, t0, {**timing, "gpu/hints": " | ".join(hints)})
    run.log_images(render_all(ev.fr, ev.rows, psi_pred, scal_pred, RESULTS / "figures", tag))
    run.finish()


def visualize(tag: str, cache_root: Path = CACHE_ROOT, holdout: int = 200, score_shots: int = 40) -> None:
    """Render and log figures for a saved model (results/<tag> dir or results/<tag>.joblib) as a `<tag>-viz` run."""
    ev = HeldOut.load(cache_root, holdout, score_shots)
    m = load_model(RESULTS / tag if (RESULTS / tag).is_dir() else RESULTS / f"{tag}.joblib")
    psi_pred, scal_pred = m.predict(ev.fr, ev.subset)
    figs = render_all(ev.fr, ev.rows, psi_pred, scal_pred, RESULTS / "figures", tag)
    run = Run(f"{tag}-viz", {"source_run": tag}, job_type="viz")
    run.log_images(figs)
    run.finish()
    logger.info(f"{tag}: figures {[str(p) for p in figs.values()]}")


COMMANDS = [prepare, prepare_thomson, baseline, train, visualize]
