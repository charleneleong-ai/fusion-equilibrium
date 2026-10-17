"""Label-free check of a built DIII-D submission: the magnetic axis the official derivation finds in each frame, and
its frame-to-frame jitter (rms second difference in time within a shot). bf16 noise in a grid member is independent
from frame to frame, so it shows up as jitter even without ground truth.

python scripts/axis_jitter.py <n_shots> <submission.npz> ...
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fusioneq.score import AXIS_SIGN, N_ITER, Scorer, derive_frame, extract_lcfs_with_sign

SC = Scorer("DIII-D")


def shot_axis(psi: np.ndarray) -> np.ndarray:
    """(T, 2) R_axis, Z_axis of each frame (NaN where the derivation fails)."""
    out = np.full((len(psi), 2), np.nan)
    sign = AXIS_SIGN["DIII-D"]
    for k, p in enumerate(np.asarray(psi, np.float64)):
        ex, _ = extract_lcfs_with_sign(
            p, SC.R, SC.Z, "DIII-D", SC.mask_coarse, SC.mask_f, n_iter=N_ITER, axis_sign=sign
        )
        v = derive_frame(p, SC.R, SC.Z, "DIII-D", SC.mask_coarse, SC.mask_f, ex, with_li=False, axis_sign=sign)
        out[k] = v["R_axis"], v["Z_axis"]
    return out


def jitter(axes: list[np.ndarray]) -> np.ndarray:
    """(2,) rms second difference in time of R_axis, Z_axis, pooled over shots [m]."""
    d2 = np.concatenate([a[2:] - 2 * a[1:-1] + a[:-2] for a in axes])
    return np.sqrt(np.nanmean(d2**2, 0))


def main(n_shots: int, files: list[Path]) -> None:
    axes = {}
    for f in files:
        z = np.load(f)
        keys = sorted(k for k in z.files if k.endswith("_psirz"))[:n_shots]
        with ProcessPoolExecutor(6) as ex:
            axes[f.parent.name] = list(ex.map(shot_axis, [z[k] for k in keys]))
        j = jitter(axes[f.parent.name])
        print(f"{f.parent.name}: {n_shots} shots, axis jitter R {1e3 * j[0]:.3f} mm, Z {1e3 * j[1]:.3f} mm", flush=True)
    names = list(axes)
    for other in names[1:]:
        d = np.concatenate([b - a for a, b in zip(axes[names[0]], axes[other], strict=True)])
        rms = 1e3 * np.sqrt(np.nanmean(d**2, 0))
        print(f"{other} - {names[0]}: axis shift rms R {rms[0]:.3f} mm, Z {rms[1]:.3f} mm", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
