"""Two head sets blended into the map reading, target = map + w1 (h1 - map) + w2 (h2 - map), (w1, w2) per scalar
chosen on the validation group only. argv: h1 names (comma-separated, averaged) h2 names. Writes target_<h1>__<h2>."""

import json
import sys

import numpy as np
from blend_eval import r2
from fincommon import OUT, Data

from fusioneq.shape_map import CONS_INDEX

WS = np.round(np.linspace(0, 1, 21), 2)


def main() -> None:
    d = Data()
    h1, h2 = (np.mean([np.load(OUT / f"heads_{n}.npy") for n in a.split(",")], 0) for a in sys.argv[1:3])
    v = d.sel("validation")
    w = {}
    for k, j in CONS_INDEX.items():
        y, m, a, b = d.label[v, j], d.pred[v, j], h1[v, j] - d.pred[v, j], h2[v, j] - d.pred[v, j]
        grid = [(x1, x2) for x1 in WS for x2 in WS if x1 + x2 <= 1]
        w[k] = max(grid, key=lambda g: r2(y, m + g[0] * a + g[1] * b))
    print("w", w)
    w1, w2 = (np.array([w[k][i] for k in CONS_INDEX]) for i in (0, 1))
    target = d.pred + w1 * (h1 - d.pred) + w2 * (h2 - d.pred)
    for g in d.groups:
        s = d.sel(g)
        r = [r2(d.label[s, j], target[s, j]) for j in range(7)]
        print(g, np.round(r, 4).tolist(), f"mean {np.mean(r):.5f}")
    name = f"{sys.argv[1].replace(',', '+')}__{sys.argv[2].replace(',', '+')}"
    np.save(OUT / f"target_{name}.npy", target)
    (OUT / f"target_{name}.json").write_text(json.dumps(w))


if __name__ == "__main__":
    main()
