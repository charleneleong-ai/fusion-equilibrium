"""Finish the Final pick on all 200 held-out shots toward a target file, then score the 160 never-scored shots and
the five groups against the unfinished map. argv: variant target ops (ops: all | shape | li | axis+li ...)."""

import json
import sys
import time

import numpy as np
from fincommon import OUT, SCALE, Data, brief, log

from fusioneq.finish import CONS_INDEX, Finisher

OPS = {"axis": ["R_axis", "Z_axis"], "shape": ["kappa", "tri_top", "tri_bot", "volume"], "li": ["li"]}


def pick(r: dict) -> dict:
    return {k: r[k] for k in ("S", "r2_psi", "dlcfs", "consistency", "r2_cons_each")}


def main() -> None:
    variant, target_name, ops = sys.argv[1], sys.argv[2], sys.argv[3]
    kv = dict(a.split("=") for a in sys.argv[4:])
    conv = {"passes": int, "scales": lambda x: tuple(float(y) for y in x.split(","))}
    kv = {k: conv.get(k, float)(v) for k, v in kv.items()}
    names = [n for o in (OPS if ops == "all" else ops.split("+")) for n in OPS[o]]
    d = Data()
    target = np.load(OUT / f"target_{target_name}.npy")
    keep = np.isin(np.arange(7), [CONS_INDEX[n] for n in names]) & (np.abs(target - d.pred) > 0).any(0)
    target = np.where(keep, target, np.nan)
    t0 = time.time()
    fin = Finisher(SCALE, shape=any(n in names for n in OPS["axis"] + OPS["shape"]), li="li" in names, **kv)
    psi, v, ok = fin.many(np.asarray(d.psi), target, 14)
    took = time.time() - t0
    log(f"{variant} {target_name} {ops} {kv}: finished {len(psi)} frames in {took:.0f}s, accepted {ok.mean():.3f}")
    base_path = OUT / "baseline_groups.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    res = {}
    for g in d.groups:
        s = d.sel(g)
        if g not in base:
            base[g] = pick(d.score(g, np.asarray(d.psi[s])))
            base_path.write_text(json.dumps(base, indent=1, default=float))
        r = d.score(g, psi[s])
        res[g] = {"dS": r["S"] - base[g]["S"], "dr2psi": r["r2_psi"] - base[g]["r2_psi"],
                  "ddlcfs": r["dlcfs"] - base[g]["dlcfs"], "dcons": r["consistency"] - base[g]["consistency"],
                  "S": r["S"], "cons_each": r["r2_cons_each"]}  # fmt: skip
        log(f"{variant} {g}: dS {res[g]['dS']:+.6f} dr2psi {res[g]['dr2psi']:+.2e} | base {brief(base[g])}")
        log(f"{variant} {g}: finished {brief(r)}")
    (OUT / f"finish_{variant}.json").write_text(json.dumps(res, indent=1, default=float))
    log("DONE")


if __name__ == "__main__":
    main()
