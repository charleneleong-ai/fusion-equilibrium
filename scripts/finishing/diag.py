"""Validation-group diagnostic of one target file: per scalar, how much of the intended move (target - reading) the
operators realise, with and without the gate, and the per-scalar R2 before / after. argv: target ops [gate]."""

import sys

import numpy as np
from fincommon import OUT, SCALE, Data, brief, log

from fusioneq.finish import CONS_INDEX, Finisher

OPS = {"axis": ["R_axis", "Z_axis"], "shape": ["kappa", "tri_top", "tri_bot", "volume"], "li": ["li"]}


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def main() -> None:
    name, ops = sys.argv[1], sys.argv[2]
    kv = dict(a.split("=") for a in sys.argv[3:])
    gate = bool(float(kv.pop("gate", 1)))
    conv = {"passes": int, "scales": lambda x: tuple(float(y) for y in x.split(","))}
    kv = {k: conv.get(k, float)(v) for k, v in kv.items()}
    names = [n for o in (OPS if ops == "all" else ops.split("+")) for n in OPS[o]]
    d = Data()
    s = d.sel("validation")
    psi, v0, y = np.asarray(d.psi[s]), d.pred[s], d.label[s]
    target = np.load(OUT / f"target_{name}.npy")[s]
    keep = np.isin(np.arange(7), [CONS_INDEX[n] for n in names]) & (np.abs(target - v0) > 0).any(0)
    target = np.where(keep, target, np.nan)
    fin = Finisher(SCALE, shape=any(n not in ("li",) for n in names), li="li" in names, gate=gate, **kv)
    out, v1, ok = fin.many(psi, target, 14)
    log(f"{name} {ops} gate {gate} {kv}: accepted {ok.mean():.3f}")
    for k, j in CONS_INDEX.items():
        want, got = target[:, j] - v0[:, j], v1[:, j] - v0[:, j]
        f = np.isfinite(want) & np.isfinite(got) & ok
        slope = float((want[f] * got[f]).sum() / max((want[f] ** 2).sum(), 1e-30)) if f.any() else float("nan")
        log(f"  {k:8} realised slope {slope:+.3f} R2 {r2(y[:, j], v0[:, j]):.4f} -> {r2(y[:, j], v1[:, j]):.4f}"
            f" (target {r2(y[:, j], np.where(np.isfinite(target[:, j]), target[:, j], v0[:, j])):.4f})")  # fmt: skip
    log("base", brief(d.score("validation", psi)))
    log("fin ", brief(d.score("validation", out)))


if __name__ == "__main__":
    main()
