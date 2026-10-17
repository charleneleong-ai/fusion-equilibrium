"""Does build precision explain a4b4092c? The 160-shot screen scores grid shapes predicted on the CPU in float32, but
07cfd3ff was built on the CPU (float32) and a4b4092c on the GPU, where GridPredictor ran under bf16 autocast; both
submissions store psi as float16. This re-predicts the four grids under emulated bf16 autocast (on the CPU), rebuilds
H and H + grid_fno as shipped, and scores them frame by frame like the screen (docs/fno-transfer-diagnosis.md).

Needs framewise.py and cons_by_family.py (feat/fusion-eq-cons-analysis; ~/fusion/cons on the box) on PYTHONPATH.
python scripts/fno_precision.py <out_dir>
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from cons_by_family import H_GRIDS, SCALARS, C, Components, groups, shot_inputs
from framewise import frame_scores

from fusioneq.evaluation import HeldOut
from fusioneq.grid import GridPredictor
from fusioneq.score import Scorer
from fusioneq.settings import RESULTS
from fusioneq.targets import envelope

GRIDS = ("grid_v2", "grid_v2_s1", "grid_v3", "grid_fno")
REFERENCE = Path.home() / "fusion" / "cons" / "fs"  # H.pkl / H_fno.pkl: the screen's float32 frame scores
SUBMISSIONS = {"H": "submission_v21_27_grids3", "H_fno": "submission_v21_27_grids3_fno"}


def bf16_round(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def round_like_autocast(net: torch.nn.Module) -> None:
    """Emulate CUDA bf16 autocast in float32: every conv / linear sees bf16-rounded inputs and weights and emits a
    bf16-rounded output (the spectral layers already run in float32). CPU autocast is exact but ~13x slower for the
    U-Nets on a host without native bf16; `check` compares the two on a sample."""
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d | torch.nn.Linear):
            with torch.no_grad():
                mod.weight.copy_(bf16_round(mod.weight))
                if mod.bias is not None:
                    mod.bias.copy_(bf16_round(mod.bias))
            mod.register_forward_pre_hook(lambda _m, args: tuple(bf16_round(a) for a in args))
            mod.register_forward_hook(lambda _m, _args, out: bf16_round(out))


def bf16_shapes(ev: HeldOut, out: Path) -> None:
    held = np.nonzero(ev.held)[0]
    check = np.zeros_like(ev.held)
    check[held[:: len(held) // 128][:128]] = True
    inside = envelope().reshape(65, 65)
    for tag in GRIDS:
        path = out / f"{tag}_bf16.npy"
        if path.exists():
            continue
        m = GridPredictor.load(RESULTS / tag, device="cpu")
        fp32, autocast = m.predict_shape(ev.fr, check), m.predict_shape(ev.fr, check, bf16=True)
        round_like_autocast(m.net)
        emulated = m.predict_shape(ev.fr, check)
        rms = [float(np.sqrt(np.mean((x - fp32)[:, inside] ** 2))) for x in (autocast, emulated)]
        print(f"{tag}: rms vs float32 on 128 frames, CPU autocast {rms[0]:.2e}, emulated {rms[1]:.2e}", flush=True)
        np.save(path, m.predict_shape(ev.fr, ev.held))
        print(f"{tag}: bf16 shapes saved", flush=True)


def configs(comps: Components) -> dict[str, np.ndarray]:
    bf = tuple(g.replace("_fp32", "_bf16") for g in H_GRIDS)
    shipped = {
        "H_f16": comps.h(),
        "H_fno_f16": comps.blend(0.10, 0.05, 0.15, 0.70, (*H_GRIDS, "grid_fno_fp32")),
        "H_bf16_f16": comps.blend(0.10, 0.10, 0.15, 0.65, bf),
        "H_fno_bf16_f16": comps.blend(0.10, 0.05, 0.15, 0.70, (*bf, "grid_fno_bf16")),
    }
    return {k: v.astype(np.float16).astype(np.float32) for k, v in shipped.items()}


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    inside = envelope().reshape(65, 65)
    return float(np.sqrt(np.mean((a[:, inside].astype(np.float64) - b[:, inside]) ** 2)))


def test_fingerprint(n_shots: int = 120) -> None:
    """Envelope rms of a4b4092c - 07cfd3ff per frame on the public test file, to set beside the held-out builds."""
    home = Path.home() / "fusion"
    a, b = (np.load(home / d / "diii_d_public_test.npz") for d in SUBMISSIONS.values())
    keys = [k for k in a.files if k.endswith("_psirz")][:n_shots]
    d = np.mean([rms_diff(b[k][...].astype(np.float32), a[k][...].astype(np.float32)) ** 2 for k in keys])
    print(f"test: rms(a4b4092c - 07cfd3ff) inside the envelope, {len(keys)} shots: {np.sqrt(d):.3e}", flush=True)


def main(out: Path) -> None:
    torch.set_num_threads(8)
    out.mkdir(parents=True, exist_ok=True)
    ev, rows, sc = HeldOut.load(), np.load(C / "rows.npy"), Scorer("DIII-D")
    bf16_shapes(ev, out)
    comps = Components()
    comps.raw |= {f"{g}_bf16": np.load(out / f"{g}_bf16.npy", mmap_mode="r") for g in GRIDS}
    for g in GRIDS:
        print(f"{g}: rms(bf16 - fp32) {rms_diff(comps.f32(g + '_bf16'), comps.f32(g + '_fp32')):.3e}", flush=True)
    scal = np.asarray(comps.raw["scal"])
    gts, _ = shot_inputs(ev, rows, comps.base, scal)
    refs = sc.references(gts, cache_dir=REFERENCE / "refs", workers=12)
    built = configs(comps)
    print(
        f"held-out: rms(H_fno - H) f32 {rms_diff(built['H_fno_f16'], built['H_f16']):.3e}, as shipped "
        f"{rms_diff(built['H_fno_bf16_f16'], built['H_f16']):.3e}",
        flush=True,
    )
    test_fingerprint()
    fs = {k: pickle.loads((REFERENCE / f"{k}.pkl").read_bytes()) for k in ("H", "H_fno")}
    for name, psi in built.items():
        path = out / f"{name}.pkl"
        if not path.exists():
            _, preds = shot_inputs(ev, rows, psi, scal)
            path.write_bytes(pickle.dumps(frame_scores(sc, gts, refs, preds, workers=12)))
        fs[name] = pickle.loads(path.read_bytes())
        print(f"scored {name}", flush=True)
    report(fs, groups(ev, rows))


def report(fs: dict, masks: dict[str, np.ndarray]) -> None:
    pairs = {
        "screen (fp32)": ("H", "H_fno"),
        "f16 out": ("H_f16", "H_fno_f16"),
        "as shipped": ("H_f16", "H_fno_bf16_f16"),
        "both bf16": ("H_bf16_f16", "H_fno_bf16_f16"),
    }
    for label, (h, f) in pairs.items():
        cells = []
        for g, m in masks.items():
            a, b = fs[h].summary(m), fs[f].summary(m)
            cells.append(f"{g} {1e6 * (b['S'] - a['S']):+.0f}")
        print(f"dS x1e-6 {label:14}: " + "  ".join(cells), flush=True)
    m = masks["choose160"]
    for name, f in fs.items():
        s = f.summary(m)
        each = " ".join(f"{k} {s['r2_cons_each'][k]:.5f}" for k in SCALARS)
        print(
            f"{name:15} S {s['S']:.6f} psi {s['r2_psi']:.7f} dl {s['dlcfs']:.6f} cons {s['consistency']:.6f} | {each}"
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
