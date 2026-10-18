"""Target quality before any deformation: pooled R2 vs the label scalars of the map's reading, the heads, and the
per-scalar blend map + w (head - map) with w chosen on the validation group only. argv: head names (averaged)."""

import json
import sys

import numpy as np
from fincommon import OUT, Data

from fusioneq.shape_map import CONS_INDEX

WS = np.round(np.linspace(0, 1, 21), 2)


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    names = sys.argv[1:]
    d = Data()
    head = np.mean([np.load(OUT / f"heads_{n}.npy") for n in names], 0)
    v = d.sel("validation")
    w = {k: float(WS[np.argmax([r2(d.label[v, j], (d.pred + x * (head - d.pred))[v, j]) for x in WS])])
         for k, j in CONS_INDEX.items()}  # fmt: skip
    print("w", w)
    wv = np.array([w[k] for k in CONS_INDEX])
    target = d.pred + wv * (head - d.pred)
    for g in d.groups:
        s = d.sel(g)
        rows = {lab: [r2(d.label[s, j], p[s, j]) for j in range(7)] for lab, p in
                (("map", d.pred), ("head", head), ("blend", target))}  # fmt: skip
        print(g, " ".join(f"{lab} {np.round(r, 4).tolist()} mean {np.mean(r):.5f}" for lab, r in rows.items()))
    np.save(OUT / f"target_{'+'.join(names)}.npy", target)
    (OUT / f"target_{'+'.join(names)}.json").write_text(json.dumps(w))


if __name__ == "__main__":
    main()
