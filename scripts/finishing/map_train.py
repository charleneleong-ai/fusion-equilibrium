"""Residual-heads input: the Final pick's own psi (results/blend_v21_16_grids3_fno2, float32, TF32 off) on every 4th
training shot, and its scorer reading. The members were trained on these frames, so the readings are in-sample.
argv[1] = shot stride (default 4). Writes ~/fusion/finishing/map_train_s<k>.npz (rows, reading); the psi is not kept.
First checks one held-out chunk against final_psi.npy (the cached-component reproduction)."""

import sys
import time

import numpy as np
import torch
from fincommon import OUT, log

from fusioneq import grid
from fusioneq.evaluation import HeldOut
from fusioneq.finish import read_many
from fusioneq.loading import load_model
from fusioneq.settings import CACHE_ROOT, RESULTS

SHOTS_PER_CHUNK = 20
grid.EVAL_BATCH = 128  # the shared GPU had ~14 GB free; 512 ran out of memory


def main() -> None:
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ev = HeldOut.load()
    sid = np.asarray(ev.fr.shot_index)
    blend = load_model(RESULTS / "blend_v21_16_grids3_fno2")
    held_rows = np.load(CACHE_ROOT / "refiner" / "components" / "rows.npy")
    check = np.isin(sid, np.unique(sid[held_rows])[:5])
    got = blend.predict(ev.fr, check)[0]
    want = np.load(OUT / "final_psi.npy", mmap_mode="r")[np.isin(held_rows, np.nonzero(check)[0])]
    log(f"held-out check: max |diff| {np.abs(got - want).max():.2e} (psi scale {np.abs(want).max():.2f})")
    shots = np.unique(sid[ev.train])[::stride]
    rows = np.nonzero(np.isin(sid, shots))[0]
    psi, t0 = np.empty((len(rows), *want.shape[1:]), np.float32), time.time()
    chunks = np.array_split(shots, max(1, len(shots) // SHOTS_PER_CHUNK))
    for i, c in enumerate(chunks, 1):
        mask = np.isin(sid, c)
        psi[np.isin(rows, np.nonzero(mask)[0])] = blend.predict(ev.fr, mask)[0]
        if i % 5 == 0 or i == len(chunks):
            log(f"predict {i}/{len(chunks)} chunks, {time.time() - t0:.0f}s")
    log(f"{len(rows)} frames predicted in {time.time() - t0:.0f}s; reading")
    t1 = time.time()
    reading = read_many(psi, 14)
    log(f"read in {time.time() - t1:.0f}s")
    np.savez(OUT / f"map_train_s{stride}.npz", rows=rows, reading=reading)
    log("DONE")


if __name__ == "__main__":
    main()
