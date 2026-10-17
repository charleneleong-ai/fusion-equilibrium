"""The held-out DIII-D shots every command scores on, and the score record each run writes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from loguru import logger

from fusioneq.cache import Frames
from fusioneq.score import Scorer, summary_line
from fusioneq.settings import CACHE_ROOT, RESULTS
from fusioneq.tracking import Run

Shots = list[dict[str, np.ndarray]]


def shot_rows(fr: Frames, idx: np.ndarray, n_shots: int) -> np.ndarray:
    """Mask of every frame of the first n_shots shots touched by idx."""
    return np.isin(fr.shot_index, np.unique(fr.shot_index[idx])[:n_shots])


def shots_from_frames(fr: Frames, mask: np.ndarray, psi_pred: np.ndarray, scal_pred: np.ndarray) -> tuple[Shots, Shots]:
    """(ground truth, predictions) per shot, the scorer's input layout."""
    in_mask = fr.shot_index[mask]
    idx_all = np.nonzero(mask)[0]
    shots, preds = [], []
    for sid in np.unique(in_mask):
        local = np.nonzero(in_mask == sid)[0]
        rows = idx_all[local]
        shots.append({"psi": fr.psi[rows].astype(np.float32), "q95": fr.scalars[rows, 0], "betaN": fr.scalars[rows, 1]})
        preds.append({"psirz": psi_pred[local], "q95": scal_pred[local, 0], "betaN": scal_pred[local, 1]})
    return shots, preds


def score_frames(
    fr: Frames,
    psi: np.ndarray,
    scal: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    machine: str = "DIII-D",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Official score of predictions for frames `mask` (all by default); scalars default to zero, so q95/betaN then
    contribute nothing."""
    mask = np.ones(len(fr.shot_index), bool) if mask is None else mask
    scal = np.zeros((int(mask.sum()), 2), np.float32) if scal is None else scal
    shots, preds = shots_from_frames(fr, mask, psi, scal)
    return Scorer(machine).score(shots, preds, cache_dir=cache_dir, workers=min(12, len(shots)))


@dataclass
class HeldOut:
    """`holdout` shots kept out of training (a fixed split); every command scores the first `score_shots` of them, so
    scores stay comparable across runs. The next `score_shots` are never scored: `validation` for choosing models."""

    fr: Frames
    train: np.ndarray  # training frames
    subset: np.ndarray  # scored frames
    held: np.ndarray  # every held-out frame
    cache_root: Path
    holdout: int
    score_shots: int

    @classmethod
    def load(
        cls,
        cache_root: Path = CACHE_ROOT,
        holdout: int = 200,
        score_shots: int = 40,
        spare_holdout: bool = False,
        **extra: Any,
    ) -> Self:
        """spare_holdout also trains on the held-out shots that are never scored; the scored ones stay unseen.
        `extra` fills a subclass's own fields."""
        fr = Frames.load(cache_root / "diii_d_train")
        train, held = fr.split(holdout)
        subset = shot_rows(fr, held, score_shots)
        return cls(fr, ~subset if spare_holdout else train, subset, held, cache_root, holdout, score_shots, **extra)

    @property
    def rows(self) -> np.ndarray:
        return np.nonzero(self.subset)[0]

    @property
    def validation(self) -> np.ndarray:
        """Frames of the next `score_shots` held-out shots, which are never scored."""
        return shot_rows(self.fr, self.held & ~self.subset, self.score_shots)

    def score(self, psi: np.ndarray, scal: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict[str, Any]:
        """Official DIII-D score of predictions for `mask` (the scored shots by default)."""
        return score_frames(
            self.fr, psi, scal, self.subset if mask is None else mask, cache_dir=self.cache_root / "refs"
        )

    def record(self, tag: str, model: Any, record: dict[str, Any], run: Run, t0: float) -> dict[str, Any]:
        """Save `model` as results/<tag>, score it on the scored shots, write its score record and close the run."""
        model.save(RESULTS / tag)
        res = self.score(*model.predict(self.fr, self.subset))
        logger.info(f"{tag}: {summary_line(res)}")
        save_score(tag, record, res, run, t0)
        run.finish()
        return res


def save_score(
    tag: str, record: dict[str, Any], res: dict[str, Any], run: Run, t0: float, summary: dict[str, Any] | None = None
) -> None:
    """Write results/<tag>_score.json (read by the orchestrator and the report scripts) and the W&B summary."""
    runtime = (time.time() - t0) / 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}_score.json").write_text(
        json.dumps({"tag": tag, **record, **res, "wandb_url": run.url, "runtime_min": runtime}, indent=1)
    )
    cons = {f"cons_{k}": v for k, v in res["r2_cons_each"].items() if v is not None}
    run.summary({**res, **cons, **(summary or {})})
