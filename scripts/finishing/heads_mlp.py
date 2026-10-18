"""MLP scalar heads (fusioneq.scalar_heads) trained on cons_targets train rows; held-out predictions saved.
argv: name key=value ... (HeadConfig overrides)."""

import json
import sys
import time

import numpy as np
from fincommon import OUT, log

from fusioneq.evaluation import HeldOut
from fusioneq.scalar_heads import HeadConfig, ScalarHeads
from fusioneq.settings import CACHE_ROOT


def main() -> None:
    name, kv = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
    stride = int(kv.pop("stride", 0))  # 0: cons_targets (every 4th frame); k: every k-th from train_scalars_all
    map_file = kv.pop("map", "")  # residual heads: map_train_<map>.npz, the Final pick's in-sample training readings
    cfg = HeadConfig(**{k: type(getattr(HeadConfig(), k))(json.loads(v)) for k, v in kv.items()})
    ev = HeldOut.load()
    z = np.load(CACHE_ROOT / "diii_d_train" / "cons_targets.npz")
    tr, Y = z["train_rows"], z["train"]
    if stride:
        a = np.load(OUT / "train_scalars_all.npz")
        tr, Y = a["rows"][::stride], a["y"][::stride]
    reading = held_reading = None
    if map_file:
        proxy = map_file == "proxy"  # kfold_proxy.py: out-of-sample proxy-map readings, train and held-out
        m = np.load(OUT / ("proxy_train.npz" if proxy else f"map_train_{map_file}.npz"))
        a = np.load(OUT / "train_scalars_all.npz")
        keep = np.arange(len(m["rows"]))[:: max(stride, 1)]
        tr, reading = m["rows"][keep], m["reading"][keep]
        Y = a["y"][np.searchsorted(a["rows"], tr)]
        held_reading = np.load(OUT / "proxy_held.npz")["reading"] if proxy else np.load(OUT / "heldout.npz")["pred"]
    assert ev.train[tr].all()
    held = np.load(CACHE_ROOT / "refiner" / "components" / "rows.npy")
    t0 = time.time()
    h = ScalarHeads(cfg).fit(ev.fr, tr, Y, reading)
    log(f"{name} trained on {len(tr)} frames in {time.time() - t0:.0f}s")
    h.save(OUT / f"heads_{name}")
    np.save(OUT / f"heads_{name}.npy", h.predict(ev.fr, held, held_reading))
    log("DONE")


if __name__ == "__main__":
    main()
