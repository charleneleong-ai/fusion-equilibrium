"""Finishing screen, step 1: the Final pick (B15/S1/v3+fno+fno1@0.75, subsets160_fno.py's loading) as float32 psi on
all 200 held-out shots, its scorer-read scalars, and the label scalars. Writes ~/fusion/finishing/heldout.npz."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("~/fusion/final_budget").expanduser()))
import subsets160_fno as sb  # noqa: E402

from fusioneq.consistency import scorer_scalars  # noqa: E402
from fusioneq.evaluation import HeldOut  # noqa: E402
from fusioneq.finish import read_many  # noqa: E402
from fusioneq.targets import match_envelope, shape_stats  # noqa: E402

OUT = Path("~/fusion/finishing").expanduser()
W = 16


def final_psi(rows: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly score_group's psi for config (B15, S1, v3+fno+fno1, 0.75)."""
    s = np.isin(rows, np.nonzero(mask)[0])
    trees_full = np.load(sb.C / "trees_full.npy", mmap_mode="r")
    st21 = np.asarray(np.load(sb.C / "st21_full.npy", mmap_mode="r")[s], np.float32)
    base = (0.9 * sb.mean_of(sb.BASE["B15"], "psi", s) + 0.1 * trees_full[s]).astype(np.float32)
    scal = sb.mean_of(sb.BASE["B15"], "scal", s)
    st = shape_stats(base.reshape(len(base), -1))
    trees = match_envelope(st21, st)
    hyb = match_envelope(sb.mean_of(sb.SHAPE["S1"], "psi", s), st)
    g = [np.asarray(np.load(sb.GRID_FILES[x], mmap_mode="r")[s], np.float32) for x in ("v3", "fno", "fno1")]
    grid = match_envelope(np.mean(g, axis=0), st)
    w = 0.75
    return (0.1 * base + (0.75 - w) * hyb + 0.15 * trees + w * grid).astype(np.float32), scal


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ev = HeldOut.load()
    rows = np.load(sb.C / "rows.npy")
    psi, scal = final_psi(rows, ev.held)
    np.save(OUT / "final_psi.npy", psi)
    np.save(OUT / "final_scal.npy", scal)
    choose = ev.held & ~ev.subset
    sub = np.isin(rows, np.nonzero(choose)[0])
    print(time.strftime("%H:%M:%S"), "choose160 S", ev.score(psi[sub], scal[sub], mask=choose)["S"], flush=True)
    pred = read_many(psi, W)
    print(time.strftime("%H:%M:%S"), "pred read", flush=True)
    with ProcessPoolExecutor(W) as ex:
        label = np.stack(list(ex.map(scorer_scalars, (ev.fr.psi[r] for r in rows), chunksize=256)))
    np.savez(OUT / "heldout.npz", rows=rows, pred=pred, label=label)
    print(time.strftime("%H:%M:%S"), "DONE", flush=True)


if __name__ == "__main__":
    main()
