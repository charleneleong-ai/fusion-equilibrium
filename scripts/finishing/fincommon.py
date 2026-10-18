"""Shared loading for the finishing screen: the Final pick's held-out psi, readings, labels and the five groups exactly
as subsets160_fno.py defines them."""

import time
from pathlib import Path

import numpy as np

from fusioneq.evaluation import HeldOut

OUT = Path("~/fusion/finishing").expanduser()
SCALE = np.array([0.0398, 0.0406, 0.0985, 0.2425, 0.2177, 1.298, 0.2531])  # label spread on training frames


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def groups(ev: HeldOut) -> dict[str, np.ndarray]:
    spare = ev.held & ~ev.subset & ~ev.validation
    shots = np.unique(ev.fr.shot_index[spare])
    g = {"choose160": ev.held & ~ev.subset}
    g |= {f"spare{i}": spare & np.isin(ev.fr.shot_index, p) for i, p in enumerate(np.array_split(shots, 3))}
    return g | {"validation": ev.validation, "scoring": ev.subset}


class Data:
    def __init__(self) -> None:
        self.ev = HeldOut.load()
        z = np.load(OUT / "heldout.npz")
        self.rows, self.pred, self.label = z["rows"], z["pred"], z["label"]
        self.psi = np.load(OUT / "final_psi.npy", mmap_mode="r")
        self.scal = np.load(OUT / "final_scal.npy")
        self.groups = groups(self.ev)

    def sel(self, group: str) -> np.ndarray:
        return np.isin(self.rows, np.nonzero(self.groups[group])[0])

    def score(self, group: str, psi: np.ndarray) -> dict:
        """psi for the group's rows (in rows order)."""
        s = self.sel(group)
        return self.ev.score(psi, self.scal[s], mask=self.groups[group])


def brief(r: dict) -> str:
    c = {k: round(v, 4) for k, v in r["r2_cons_each"].items()}
    return f"S {r['S']:.6f} r2psi {r['r2_psi']:.6f} dlcfs {r['dlcfs']:.5f} cons {r['consistency']:.5f} {c}"
