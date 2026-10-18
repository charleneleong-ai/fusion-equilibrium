"""The scorer's seven scalars of the label flux on EVERY training frame (cons_targets.npz has every 4th), for heads
trained on 4x the frames. Training frames only. Writes ~/fusion/finishing/train_scalars_all.npz."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fincommon import OUT, log

from fusioneq.consistency import scorer_scalars
from fusioneq.evaluation import HeldOut
from fusioneq.settings import CACHE_ROOT


def main() -> None:
    ev = HeldOut.load()
    rows = np.nonzero(ev.train)[0]
    z = np.load(CACHE_ROOT / "diii_d_train" / "cons_targets.npz")
    y = np.full((len(rows), 7), np.nan)
    have = np.searchsorted(rows, z["train_rows"])
    y[have] = z["train"]
    todo = np.setdiff1d(np.arange(len(rows)), have)
    log(f"{len(todo)} frames to derive")
    with ProcessPoolExecutor(12) as ex:
        y[todo] = np.stack(list(ex.map(scorer_scalars, (ev.fr.psi[r] for r in rows[todo]), chunksize=512)))
    np.savez(OUT / "train_scalars_all.npz", rows=rows, y=y)
    log("DONE")


if __name__ == "__main__":
    main()
