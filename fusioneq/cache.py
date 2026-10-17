"""Read back what prepare.py wrote, with a shot-level split helper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from fusioneq.data import STATIONS

HISTORY_WINDOWS = (10, 25, 50)  # past frames of 20 ms: 0.2, 0.5 and 1 s
RAMP_FRAMES = 5  # |Ip| ramp over the last 0.1 s
HISTORY_COLUMNS = ("t_shot", "ip_0.2s", "ip_0.5s", "ip_1s", "ip_ramp_0.1s", "te_core_0.2s", "te_core_0.5s")
N_CORE = len(STATIONS["DIII-D"]["core"])  # core log Te stations lead the thomson features (same count on MAST)


def shot_bounds(shot_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends): each shot's first row and one past its last; frames must be shot-contiguous and in order."""
    if np.any(np.diff(shot_index) < 0):
        raise ValueError("each shot's frames must be contiguous and in order")
    starts = np.r_[0, np.flatnonzero(np.diff(shot_index)) + 1]
    return starts, np.r_[starts[1:], len(shot_index)]


def ramp_frames(shot_index: np.ndarray, times: np.ndarray, ip: np.ndarray, edge: float = 0.1) -> np.ndarray:
    """Ramp-up / ramp-down frames: the first or last `edge` of each shot's time span, or |Ip| below 80% of the shot's
    peak. Where most of the consistency error sits (docs/diii-d-diagnosis.md)."""
    t, ip = np.asarray(times, np.float64), np.abs(np.asarray(ip, np.float64))
    out = np.zeros(len(shot_index), bool)
    for s, e in zip(*shot_bounds(shot_index), strict=True):
        phase = (t[s:e] - t[s]) / max(t[e - 1] - t[s], 1e-9)
        out[s:e] = (phase < edge) | (phase > 1 - edge) | (ip[s:e] < 0.8 * ip[s:e].max())
    return out


def shot_first(shot_index: np.ndarray) -> np.ndarray:
    """(N,) row of each frame's shot's first frame."""
    starts, ends = shot_bounds(shot_index)
    return np.repeat(starts, ends - starts)


def past_means(v: np.ndarray, first: np.ndarray, windows: tuple[int, ...]) -> list[np.ndarray]:
    """Mean of v over each frame's last `w` frames within its shot (fewer at the shot's start), per window."""
    i = np.arange(len(v))
    cs = np.r_[0.0, np.cumsum(v, dtype=np.float64)]
    out = []
    for w in windows:
        lo = np.maximum(first, i - w + 1)
        out.append((cs[i + 1] - cs[lo]) / (i + 1 - lo))
    return out


@dataclass
class Frames:
    currents: np.ndarray
    ip: np.ndarray
    thomson: np.ndarray
    psi_vac: np.ndarray
    shot_index: np.ndarray
    efit_times: np.ndarray
    grid_R: np.ndarray
    grid_Z: np.ndarray
    shots: list[dict]
    psi: np.ndarray | None = None
    scalars: np.ndarray | None = None
    thomson_hr: np.ndarray | None = None  # full-resolution, time-averaged profiles (prepare-thomson)

    @classmethod
    def load(cls, cache_dir: Path, mmap: bool = True) -> Frames:
        mode = "r" if mmap else None

        def rd(name: str) -> np.ndarray | None:
            p = cache_dir / f"{name}.npy"
            return np.load(p, mmap_mode=mode) if p.exists() else None

        return cls(
            rd("currents"),
            rd("ip"),
            rd("thomson"),
            rd("psi_vac"),
            rd("shot_index"),
            rd("efit_times"),
            rd("grid_R"),
            rd("grid_Z"),
            json.loads((cache_dir / "shots.json").read_text()),
            rd("psi"),
            rd("scalars"),
            rd("thomson_hr"),
        )

    @property
    def n_shots(self) -> int:
        return len(self.shots)

    def features(self, rows: np.ndarray | None = None, thomson_hr: bool = False) -> np.ndarray:
        """(N, F) model inputs for `rows` (all frames by default): coil currents, Ip, Thomson profiles."""
        sel = slice(None) if rows is None else rows
        if thomson_hr and self.thomson_hr is None:
            raise FileNotFoundError("thomson_hr.npy missing from this cache: run `fusioneq.cli prepare-thomson`")
        thomson = self.thomson_hr if thomson_hr else self.thomson
        return np.concatenate([self.currents[sel], self.ip[sel, None], thomson[sel]], axis=1).astype(np.float32)

    @cached_property
    def history(self) -> np.ndarray:
        """(N, len(HISTORY_COLUMNS)) past-looking context the current profile depends on (it diffuses over ~1 s)."""
        first = shot_first(np.asarray(self.shot_index))
        ip = np.abs(np.asarray(self.ip)).astype(np.float64)
        te = np.asarray(self.thomson[:, :N_CORE]).max(1).astype(np.float64)
        cols = [np.asarray(self.efit_times, np.float64) / 1e3, *past_means(ip, first, HISTORY_WINDOWS)]
        cols.append(ip - ip[np.maximum(first, np.arange(len(ip)) - RAMP_FRAMES)])
        cols += past_means(te, first, HISTORY_WINDOWS[:2])
        return np.stack(cols, 1).astype(np.float32)

    def split(self, holdout: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Boolean frame masks (train, holdout) for a shot-level split; holdout = number of shots."""
        rng = np.random.default_rng(seed)
        held = set(rng.choice(self.n_shots, holdout, replace=False).tolist())
        is_held = np.isin(self.shot_index, list(held))
        return ~is_held, is_held
