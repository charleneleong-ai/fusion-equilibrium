"""Operator ceiling on the validation group: finish toward the TRUE scalars (labels as targets) with each operator
alone and all together. Tells how much of a perfect head the deformations realise and what they cost in R2psi."""

import json
import time

import numpy as np
from fincommon import OUT, SCALE, Data, brief, log

from fusioneq.finish import CONS_INDEX, Finisher

AXIS, SHAPE_ONLY = ["R_axis", "Z_axis"], ["kappa", "tri_top", "tri_bot", "volume"]


def only(target: np.ndarray, names: list[str]) -> np.ndarray:
    keep = np.zeros(7, bool)
    keep[[CONS_INDEX[n] for n in names]] = True
    return np.where(keep, target, np.nan)


def main() -> None:
    d = Data()
    s = d.sel("validation")
    psi, label = np.asarray(d.psi[s]), d.label[s]
    base = d.score("validation", psi)
    log("base", brief(base))
    res = {"base": base["S"]}
    variants = {
        "axis": (only(label, AXIS), True, False),
        "shape": (only(label, SHAPE_ONLY), True, False),
        "li": (only(label, ["li"]), False, True),
        "all": (label, True, True),
    }
    for name, (target, shape, li) in variants.items():
        t0 = time.time()
        out, v, ok = Finisher(SCALE, shape=shape, li=li).many(psi, target, 14)
        r = d.score("validation", out)
        res[name] = r["S"]
        log(f"{name}: accepted {ok.mean():.3f} {time.time() - t0:.0f}s dS {r['S'] - base['S']:+.6f}", brief(r))
    (OUT / "oracle.json").write_text(json.dumps(res, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
