"""Residual heads, honest variant: a cheap proxy map whose readings are out-of-sample on the training frames.

Two proxy MLPs (one per half of the training shots, float32, no AMP) each predict the other half; the scorer reads
the every-4th-shot training frames from the model that never saw them, and the held-out frames from the mean of the
two models' psi. Writes ~/fusion/finishing/proxy_{train,held}.npz (rows, reading)."""

import time

import numpy as np
import torch
from fincommon import OUT, log

from fusioneq.evaluation import HeldOut
from fusioneq.finish import read_many
from fusioneq.model import NetConfig, NeuralPredictor
from fusioneq.settings import CACHE_ROOT

CFG = NetConfig(n_components=128, hidden=1024, layers=6, epochs=40, context=4, coef_loss="mix", amp=False,
                thomson_hr=True, history=True)  # fmt: skip
CHUNK = 50_000


def predict(model: NeuralPredictor, ev: HeldOut, rows: np.ndarray) -> np.ndarray:
    out = np.empty((len(rows), 65, 65), np.float32)
    for i in range(0, len(rows), CHUNK):
        mask = np.zeros(len(ev.fr.shot_index), bool)
        mask[rows[i : i + CHUNK]] = True
        out[i : i + CHUNK] = model.predict(ev.fr, mask)[0]
    return out


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ev = HeldOut.load()
    sid = np.asarray(ev.fr.shot_index)
    shots = np.unique(sid[ev.train])
    half = np.isin(shots, np.random.default_rng(2).choice(shots, len(shots) // 2, replace=False))
    read_rows = np.nonzero(np.isin(sid, shots[::4]))[0]  # the map_train_s4 frames
    held = np.load(CACHE_ROOT / "refiner" / "components" / "rows.npy")
    train_psi = np.empty((len(read_rows), 65, 65), np.float32)
    held_psi = np.zeros((len(held), 65, 65), np.float32)
    for fold in (True, False):
        t0 = time.time()
        fit_shots = shots[half == fold]
        model = NeuralPredictor(CFG).fit(ev.fr, ev.train & np.isin(sid, fit_shots))
        log(f"fold {fold}: fit on {len(fit_shots)} shots in {time.time() - t0:.0f}s")
        other = ~np.isin(sid[read_rows], fit_shots)
        train_psi[other] = predict(model, ev, read_rows[other])
        held_psi += 0.5 * predict(model, ev, held)
        log(f"fold {fold}: predicted in {time.time() - t0:.0f}s")
        del model
        torch.cuda.empty_cache()
    t0 = time.time()
    np.savez(OUT / "proxy_train.npz", rows=read_rows, reading=read_many(train_psi, 14))
    np.savez(OUT / "proxy_held.npz", rows=held, reading=read_many(held_psi, 14))
    log(f"read in {time.time() - t0:.0f}s")
    log("DONE")


if __name__ == "__main__":
    main()
