"""The official composite on held-out training shots, driven from cached frames (vendored scorer, unmodified)."""

from __future__ import annotations

import hashlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "starter" / "fusion_scoring"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from common import AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import _shape_from_contour as shape_from_contour  # noqa: E402, F401 (re-exported for fusioneq.finish)
from derive import derive_frame, internal_inductance  # noqa: E402, F401 (internal_inductance: re-exported)
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

N_POINTS, N_ITER = 512, 22
LI_IDX = CONS_SCALARS.index("li")
MASKS = {"DIII-D": VENDOR / "masks" / "d3d_envelope.npz", "MAST": VENDOR / "masks" / "mast_envelope.npz"}


def boundary_shape(r: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
    """(R0, a, kappa, delta) of one closed boundary polygon; delta averages the upper and lower triangularity."""
    r0, a = (r.max() + r.min()) / 2, (r.max() - r.min()) / 2
    kappa = (z.max() - z.min()) / (2 * a)
    delta = ((r0 - r[np.argmax(z)]) + (r0 - r[np.argmin(z)])) / (2 * a)
    return float(r0), float(a), float(kappa), float(delta)


def lcfs_shapes(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    machine: str,
    axis_sign: float,
    mask_coarse: np.ndarray | None = None,
    mask_f: np.ndarray | None = None,
) -> np.ndarray:
    """(T, 4) R0, a, kappa, delta of the LCFS the official extractor finds in each predicted frame (NaN if none);
    no envelope unless masks are passed."""
    out = np.full((len(psi), 4), np.nan)
    for k in range(len(psi)):
        c, _ = extract_lcfs_with_sign(
            np.asarray(psi[k], np.float64),
            R,
            Z,
            machine,
            mask_coarse,
            mask_f,
            n_iter=N_ITER,
            n_points=N_POINTS,
            axis_sign=axis_sign,
        )
        if c is not None:
            out[k] = boundary_shape(c[:, 0], c[:, 1])
    return out


def lcfs_shapes_parallel(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    machine: str,
    axis_sign: float,
    mask: np.ndarray | None,
    workers: int,
) -> np.ndarray:
    """lcfs_shapes over `workers` processes, inside the extraction region `mask` ((nz, nr) bool) or none."""
    mask_f = None if mask is None else mask.astype(np.float64)
    chunks = np.array_split(np.arange(len(psi)), workers * 8)
    n = len(chunks)
    with ProcessPoolExecutor(workers) as ex:
        args = ([R] * n, [Z] * n, [machine] * n, [axis_sign] * n, [mask] * n, [mask_f] * n)
        return np.concatenate(list(ex.map(lcfs_shapes, [psi[c] for c in chunks], *args)))


class Scorer:
    """Mirror of local_score.py: build references from ground truth, then score predictions shot by shot."""

    def __init__(self, machine: str = "DIII-D") -> None:
        m = np.load(MASKS[machine])
        self.machine = machine
        self.R, self.Z = m["grid_R"], m["grid_Z"]
        self.mask_coarse = m["mask_coarse"].astype(bool)
        self.mask_f = self.mask_coarse.astype(np.float64)

    def reference(self, psi_gt: np.ndarray) -> tuple[list, np.ndarray, np.ndarray, float]:
        contours, rgeos = [], []
        cons = np.full((psi_gt.shape[0], N_CONS), np.nan)
        for k in range(psi_gt.shape[0]):
            c = extract_lcfs(psi_gt[k], self.R, self.Z, self.machine, self.mask_coarse, self.mask_f, n_points=N_POINTS)
            contours.append(c)
            if c is not None:
                rgeos.append(major_radius(c))
            vals = derive_frame(psi_gt[k], self.R, self.Z, self.machine, self.mask_coarse, self.mask_f, contour=c)
            cons[k] = [vals[n] for n in CONS_SCALARS]
        return contours, cons, np.isfinite(cons), (float(np.mean(rgeos)) if rgeos else float("nan"))

    def boundary_shapes(self, psi: np.ndarray, axis_sign: float) -> np.ndarray:
        """lcfs_shapes inside this machine's scorer envelope (DIII-D only: on MAST the envelope may not shape a
        prediction, rule 3)."""
        return lcfs_shapes(psi, self.R, self.Z, self.machine, axis_sign, self.mask_coarse, self.mask_f)

    def references(self, shots: list[dict], cache_dir: Path | None = None, workers: int = 12) -> list:
        """Ground-truth LCFS + derived scalars per shot; identical across models, so cached on disk by content hash."""
        key = hashlib.sha1(b"".join(np.ascontiguousarray(s["psi"][::7]).tobytes()[:4096] for s in shots)).hexdigest()[
            :12
        ]
        path = None if cache_dir is None else cache_dir / f"refs_{self.machine}_{len(shots)}_{key}.pkl"
        if path is not None and path.exists():
            return pickle.loads(path.read_bytes())
        with ProcessPoolExecutor(workers) as ex:
            refs = list(ex.map(self._reference_np, [s["psi"].astype(np.float64) for s in shots]))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(refs))
        return refs

    def _reference_np(self, psi_gt: np.ndarray) -> tuple:
        return self.reference(psi_gt)

    def score(self, shots: list[dict], preds: list[dict], cache_dir: Path | None = None, workers: int = 12) -> dict:
        """shots: [{psi (T,65,65), q95 (T,), betaN (T,)}]; preds: same keys with psirz. Returns the leaderboard row."""
        refs = self.references(shots, cache_dir, workers)
        stats, means = self._ref_stats(shots, refs)
        totals = {s: 0.0 for s in PSI_SIGNS}
        for s, p in zip(shots, preds, strict=True):
            rr, _, _ = psi_residuals(s["psi"], p["psirz"], means[0])
            for sg in PSI_SIGNS:
                totals[sg] += rr[sg]
        psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
        acc = Accum()
        acc.psi_sign = psi_sign
        jobs = [(s, ref, p, psi_sign, means) for s, ref, p in zip(shots, refs, preds, strict=True)]
        with ProcessPoolExecutor(workers) as ex:
            for part in ex.map(self._score_job, jobs):
                acc.add(part)
        return finalize_machine(acc, stats)

    def _score_job(self, job: tuple) -> dict:
        return self._score_shot(*job)

    def _ref_stats(self, shots: list[dict], refs: list) -> tuple[dict, tuple]:
        psi_all = np.concatenate([s["psi"].astype(np.float64).ravel() for s in shots])
        psi_all = psi_all[np.isfinite(psi_all)]
        scal = [np.concatenate([np.asarray(s[n], float) for s in shots]) for n in SCALARS]
        scal = [v[np.isfinite(v)] for v in scal]
        cons = [np.concatenate([r[1][r[2][:, j], j] for r in refs]) for j in range(N_CONS)]
        stats = {
            "psi_sum": psi_all.sum(),
            "psi_sumsq": (psi_all**2).sum(),
            "psi_n": psi_all.size,
            "scal_sum": np.array([v.sum() for v in scal]),
            "scal_sumsq": np.array([(v**2).sum() for v in scal]),
            "scal_n": np.array([v.size for v in scal], float),
            "cons_sum": np.array([v.sum() for v in cons]),
            "cons_sumsq": np.array([(v**2).sum() for v in cons]),
            "cons_n": np.array([v.size for v in cons], float),
        }
        means = (
            psi_all.mean(),
            np.array([v.mean() if v.size else 0.0 for v in scal]),
            np.array([v.mean() if v.size else 0.0 for v in cons]),
        )
        return stats, means

    def _score_shot(self, gt: dict, ref: tuple, pred: dict, psi_sign: float, means: tuple) -> dict:
        mean_psi, mean_scal, mean_cons = means
        contours, cons_gt, cmask, rgeo = ref
        psi_gt = gt["psi"]
        T = psi_gt.shape[0]
        axis_sign = AXIS_SIGN[self.machine] * psi_sign
        ss_res_psi, nonfinite, _ = psi_residuals(psi_gt, pred["psirz"], mean_psi)
        ss_tot_shot = float(np.sum((psi_gt.astype(np.float64) - mean_psi) ** 2))
        shot_r2 = {s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0) for s in PSI_SIGNS}
        ss_res_scal = np.zeros(N_SCALARS)
        for j, name in enumerate(SCALARS):
            g = np.asarray(gt[name], float)
            d = np.isfinite(g)
            p = np.asarray(pred[name], float)
            pj = np.where(np.isfinite(p[d]), p[d], mean_scal[j])
            ss_res_scal[j] = float(np.sum((g[d] - pj) ** 2))
        ss_res_cons, cons_frames, cons_fails = np.zeros(N_CONS), np.zeros(N_CONS, np.int64), np.zeros(N_CONS, np.int64)
        ds, n_fail, n_frames = [], 0, 0
        psi_pred = np.asarray(pred["psirz"], np.float64)
        for k in range(T):
            ref_c, need = contours[k], cmask[k]
            if ref_c is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(
                psi_pred[k],
                self.R,
                self.Z,
                self.machine,
                self.mask_coarse,
                self.mask_f,
                n_iter=N_ITER,
                axis_sign=axis_sign,
            )
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(
                    psi_pred[k],
                    self.R,
                    self.Z,
                    self.machine,
                    self.mask_coarse,
                    self.mask_f,
                    contour=ex,
                    with_li=bool(need[LI_IDX]),
                    axis_sign=axis_sign,
                )
                for j in np.nonzero(need)[0]:
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v, cons_fails[j] = mean_cons[j], cons_fails[j] + 1
                    ss_res_cons[j] += (cons_gt[k, j] - v) ** 2
        return {
            "ss_res_psi": ss_res_psi,
            "psi_pixels": int(psi_gt.size),
            "psi_nonfinite": nonfinite,
            "ss_res_scal": ss_res_scal,
            "ss_res_cons": ss_res_cons,
            "cons_frames": cons_frames,
            "cons_fails": cons_fails,
            "shot_r2_psi": shot_r2,
            "shot_dlcfs": float(np.mean(ds)) if ds else None,
            "lcfs_frames": n_frames,
            "lcfs_fails": n_fail,
            "ref_lcfs_dropped": sum(c is None for c in contours),
            "missing": False,
        }


def psi_residuals(psi_gt: np.ndarray, pred: np.ndarray, mean_psi: float) -> tuple[dict, int, bool]:
    g = psi_gt.astype(np.float64)
    p = np.asarray(pred, np.float64)
    nf = ~np.isfinite(p)
    out = {}
    for s in PSI_SIGNS:
        q = np.where(nf, mean_psi, s * p) if nf.any() else s * p
        out[s] = float(np.sum((g - q) ** 2))
    return out, int(nf.sum()), False


def summary_line(res: dict) -> str:
    return (
        f"S={res['S']:.4f} r2psi={res['r2_psi']:.4f} r2qb={res['r2_qb']:.4f} dlcfs={res['dlcfs']:.4f} "
        f"cons={res['consistency']:.4f} sign={res['psi_sign']:+d} lcfs_fail={res['lcfs_fail_frac']:.3f} "
        f"cons_each={ {k: (round(v, 3) if v is not None else None) for k, v in res['r2_cons_each'].items()} }"
    )
