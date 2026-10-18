"""DIII-D finishing step (docs/finishing.md): deform each predicted flux map so the scorer reads target values of its
consistency scalars, the targets being blends of the map's own reading and scalar heads trained on DIII-D labels.

Two deformations, applied in order:

1. A smooth displacement u of the map, psi'(x) = psi(x - u(x)), u = sum_k p_k B_k over six basis fields: two
   Gaussian bumps at the magnetic axis (sigma SIGMA_AXIS) that move the axis in R and Z, and four fields inside a
   window around the plasma that vanish at the axis: a vertical and a radial stretch about it (kappa, volume) and an
   upper and a lower quadratic shear of R (tri_top, tri_bot). To first order the warped LCFS is the old one advected,
   C + u(C), so p is solved by Gauss-Newton on the polygon alone, without re-extracting the boundary.
2. li: a monotone remap of the flux inside the LCFS (li_shape.CoreShaper on the scorer's own masked contour), which
   leaves every flux surface, the boundary and the axis cell in place.

Each frame keeps the finished map only if the scorer's reading of it is nearer the targets than before.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits

from fusioneq.cache import Frames
from fusioneq.li_shape import CoreShaper
from fusioneq.scalar_heads import ScalarHeads
from fusioneq.score import (
    AXIS_SIGN,
    N_ITER,
    Scorer,
    derive_frame,
    extract_lcfs_with_sign,
    shape_from_contour,
    symmetric_hausdorff,
)
from fusioneq.settings import RESULTS
from fusioneq.shape_map import CONS_INDEX

SCORER = Scorer("DIII-D")
DIIID_SIGN = AXIS_SIGN["DIII-D"]
SHAPE = ("kappa", "tri_top", "tri_bot", "volume")
SIGMA_AXIS = 0.15  # m, the axis bumps' width
TAPER = (1.2, 1.7)  # window: 1 inside normalised radius TAPER[0], cosine to 0 at TAPER[1]
MAX_SHIFT = 0.03  # m, largest displacement anywhere
JUMP_TOL = 0.005  # m


def one_thread() -> None:
    """Worker initializer: one BLAS / OpenMP thread per process. Without it, workers forked after torch has run
    (the heads) each start a full thread pool, and the public-test finish ran 5x slower."""
    threadpool_limits(1)
    torch.set_num_threads(1)


def read_frame(psi: np.ndarray, with_li: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """((7,) consistency scalars in CONS_INDEX order, LCFS polygon or None) of one predicted (65, 65) DIII-D frame,
    exactly as the scorer reads a submission in DIII-D's own sign: the envelope-masked extraction with the pinned
    sign, then derive_frame on that contour."""
    p = np.asarray(psi, np.float64)
    s = SCORER
    c, _ = extract_lcfs_with_sign(p, s.R, s.Z, "DIII-D", s.mask_coarse, s.mask_f, n_iter=N_ITER, axis_sign=DIIID_SIGN)
    v = derive_frame(p, s.R, s.Z, "DIII-D", s.mask_coarse, s.mask_f, contour=c, with_li=with_li, axis_sign=DIIID_SIGN)
    return np.array([v[k] for k in CONS_INDEX]), c


def read_rows(psi: np.ndarray) -> np.ndarray:
    return np.stack([read_frame(p)[0] for p in psi])


def read_many(psi: np.ndarray, workers: int) -> np.ndarray:
    """(N, 7) read_frame scalars over `workers` processes."""
    chunks = np.array_split(np.arange(len(psi)), max(1, min(len(psi), workers * 8)))
    with ProcessPoolExecutor(workers, initializer=one_thread) as ex:
        return np.concatenate(list(ex.map(read_rows, [np.asarray(psi[c]) for c in chunks])))


def polygon_shape(c: np.ndarray) -> np.ndarray:
    """(4,) the scorer's SHAPE scalars of an (N, 2) polygon."""
    v = shape_from_contour(c)
    return np.array([v[k] for k in SHAPE])


@dataclass(frozen=True)
class Warp:
    """The six displacement fields of one frame, anchored on its axis (ra, za) and LCFS extent."""

    ra: float
    za: float
    rgeo: float
    a: float
    ztop: float  # Zmax - za
    zbot: float  # za - Zmin

    @classmethod
    def of(cls, axis: tuple[float, float], c: np.ndarray) -> Warp:
        r, z = c[:, 0], c[:, 1]
        return cls(axis[0], axis[1], (r.max() + r.min()) / 2, (r.max() - r.min()) / 2, z.max() - axis[1],
                   axis[1] - z.min())  # fmt: skip

    def fields(self, R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(uR, uZ), each (6, *R.shape): the basis displacements at points (R, Z)."""
        dr, dz = R - self.ra, Z - self.za
        g = np.exp(-(dr**2 + dz**2) / (2 * SIGMA_AXIS**2))
        h = np.where(dz > 0, self.ztop, self.zbot)
        rho = np.hypot((R - self.rgeo) / self.a, dz / h)
        t = np.clip((rho - TAPER[0]) / (TAPER[1] - TAPER[0]), 0, 1)
        w = 0.5 * (1 + np.cos(np.pi * t))
        up, lo = np.clip(dz / self.ztop, 0, None) ** 2, np.clip(-dz / self.zbot, 0, None) ** 2
        zero = np.zeros_like(R)
        uR = np.stack([g, zero, zero, dr * w, -self.a * up * w, -self.a * lo * w])
        uZ = np.stack([zero, g, dz * w, zero, zero, zero])
        return uR, uZ

    def moved(self, c: np.ndarray, p: np.ndarray) -> np.ndarray:
        """(N, 2) polygon c advected by the displacement p (einsum: Accelerate's matmul warns spuriously here)."""
        bR, bZ = self.fields(c[:, 0], c[:, 1])
        return c + np.stack([np.einsum("k,kn->n", p, bR), np.einsum("k,kn->n", p, bZ)], 1)

    def solve(self, c: np.ndarray, target: np.ndarray, iters: int = 4) -> np.ndarray:
        """(6,) p moving the axis by target[:2] and the advected polygon's SHAPE scalars onto target[2:] (Newton,
        numerical Jacobian on the polygon; the axis fields vanish on the LCFS and the shape fields at the axis)."""
        p = np.zeros(6)
        p[:2] = target[:2]

        def shape(q: np.ndarray) -> np.ndarray:
            return polygon_shape(self.moved(c, q))

        for _ in range(iters):
            f0 = shape(p)
            J = np.stack([(shape(p + e) - f0) / 1e-3 for e in 1e-3 * np.eye(6)[2:]], 1)
            p[2:] += np.linalg.lstsq(J, target[2:] - f0, rcond=None)[0]
        return p

    def apply(self, psi: np.ndarray, p: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """psi(x - u(x)) on the grid (cubic spline), u capped at MAX_SHIFT."""
        RR, ZZ = np.meshgrid(R, Z)
        bR, bZ = self.fields(RR, ZZ)
        uR, uZ = np.tensordot(p, bR, 1), np.tensordot(p, bZ, 1)
        scale = min(1.0, MAX_SHIFT / max(float(np.hypot(uR, uZ).max()), 1e-12))
        fr = (RR - scale * uR - R[0]) / (R[1] - R[0])
        fz = (ZZ - scale * uZ - Z[0]) / (Z[1] - Z[0])
        out = map_coordinates(np.asarray(psi, np.float64), [fz, fr], order=3, mode="nearest")
        return np.where(np.hypot(uR, uZ) > 0, out, psi)


SHAPER = CoreShaper(SCORER.R, SCORER.Z, "DIII-D", DIIID_SIGN, mask_coarse=SCORER.mask_coarse)
CI = [CONS_INDEX[k] for k in ("R_axis", "Z_axis", *SHAPE)]
LI = CONS_INDEX["li"]


@dataclass(frozen=True)
class Finisher:
    """Per-frame finishing toward target scalars; `scale` (7,) normalises each scalar's miss (the label spread)."""

    scale: np.ndarray
    shape: bool = True
    li: bool = True
    passes: int = 2  # warp, re-read, warp the remainder
    gate: bool = True  # keep a finished frame only if its reading is nearer the targets
    jump: float = np.inf  # reject a warp whose LCFS moved more than jump x the intended move (+ JUMP_TOL)
    scales: tuple[float, ...] = (1.0,)  # step sizes tried per pass
    weight: np.ndarray | None = None  # (7,): frame() then takes head predictions, target = v0 + weight (head - v0)

    def miss(self, v: np.ndarray, target: np.ndarray, cols: list[int] | None = None) -> float:
        d = (v - target) / (self.scale if cols is None else self.scale[cols])
        return float(np.sum(np.where(np.isfinite(d), d, 10.0) ** 2))

    def warp(self, psi: np.ndarray, v0: np.ndarray, c0: np.ndarray, target: np.ndarray) -> np.ndarray:
        """psi warped toward the axis and SHAPE targets: each pass solves the warp from the current reading, tries it
        at every one of `scales` and keeps the candidate whose reading is nearest (the scorer's boundary moves in
        grid-cell steps near X-points, so the solved step alone often over- or undershoots)."""
        t = np.where(np.isfinite(target[CI]), target[CI], v0[CI])
        best = (self.miss(v0[CI], t, CI), np.asarray(psi, np.float64), v0, c0)
        for _ in range(self.passes):
            _, cur, v, c = best
            warp = Warp.of((v[0], v[1]), c)
            p = warp.solve(c, np.r_[t[:2] - v[CI][:2], t[2:]])
            intended = float(np.hypot(*(warp.moved(c, p) - c).T).max())
            for k in self.scales:
                out = warp.apply(cur, k * p, SCORER.R, SCORER.Z)
                v1, c1 = read_frame(out, with_li=False)
                if c1 is None or symmetric_hausdorff(c1, c) > self.jump * k * intended + JUMP_TOL:
                    continue
                if (m := self.miss(v1[CI], t, CI)) < best[0]:
                    best = (m, out, v1, c1)
        return best[1]

    def frame(self, psi: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """(finished psi, its reading, accepted) for one frame and (7,) targets (NaN: leave that scalar), or with
        `weight` set, (7,) head predictions."""
        v0, c0 = read_frame(psi)
        if c0 is None:
            return psi, v0, False
        if self.weight is not None:
            target = np.where(self.weight > 0, v0 + self.weight * (target - v0), np.nan)
        out = self.warp(psi, v0, c0, target) if self.shape else np.asarray(psi, np.float64)
        if self.li and np.isfinite(target[LI]):
            bump, contour = SHAPER.bump(out)
            curve = SHAPER.curve(out, bump, contour)
            out = out + SHAPER.coefficient(curve[None], target[LI : LI + 1])[0] * bump
        v1, c1 = read_frame(out)
        ok = c1 is not None and (not self.gate or self.miss(v1, target) < self.miss(v0, target))
        return (out, v1, True) if ok else (np.asarray(psi, np.float64), v0, False)

    def rows(self, psi: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = [self.frame(p, t) for p, t in zip(psi, target, strict=True)]
        return (np.stack([o[0] for o in out]).astype(np.float32), np.stack([o[1] for o in out]),
                np.array([o[2] for o in out]))  # fmt: skip

    def many(self, psi: np.ndarray, target: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(finished psi float32, readings (N, 7), accepted (N,)) over `workers` processes."""
        chunks = np.array_split(np.arange(len(psi)), max(1, min(len(psi), workers * 8)))
        with ProcessPoolExecutor(workers, initializer=one_thread) as ex:
            parts = list(ex.map(self.rows, [np.asarray(psi[c]) for c in chunks], [target[c] for c in chunks]))
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


OPS = {"axis": ("R_axis", "Z_axis"), "shape": SHAPE, "li": ("li",)}


@dataclass(frozen=True)
class FinishSpec:
    """A committed finishing recipe (configs/finish/): the heads (HeadConfig fields, the label stride over training
    frames, the seeds averaged), the per-scalar blend weights toward them (chosen on the validation shots) and the
    operators and Finisher settings."""

    heads: dict[str, Any]
    stride: int
    seeds: tuple[int, ...]
    weights: dict[str, float]
    ops: tuple[str, ...] = ("axis", "li")
    passes: int = 2
    jump: float = 2.0
    scales: tuple[float, ...] = (1.0,)
    name: str = field(default="", compare=False)

    @classmethod
    def load(cls, path: Path) -> FinishSpec:
        d = json.loads(path.read_text())
        return cls(**{**d, "seeds": tuple(d["seeds"]), "ops": tuple(d["ops"]), "scales": tuple(d["scales"])},
                   name=path.stem)  # fmt: skip

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: v for k, v in asdict(self).items() if k != "name"}, indent=1))

    def head_tags(self) -> list[str]:
        return [f"heads_{self.name}_s{s}" for s in self.seeds]

    def weight(self) -> np.ndarray:
        """(7,) blend weight per scalar in CONS_INDEX order, 0 outside the spec's operators."""
        on = {n for op in self.ops for n in OPS[op]}
        return np.array([self.weights.get(k, 0.0) if k in on else 0.0 for k in CONS_INDEX])

    def finisher(self, scale: np.ndarray) -> Finisher:
        return Finisher(scale, shape=bool({"axis", "shape"} & set(self.ops)), li="li" in self.ops,
                        passes=self.passes, jump=self.jump, scales=self.scales, weight=self.weight())  # fmt: skip


def finish_frames(fr: Frames, psi: np.ndarray, spec: Path, workers: int) -> np.ndarray:
    """(N, 65, 65) float32: every frame of `fr` finished toward the mean of the recipe's heads."""
    s = FinishSpec.load(spec)
    heads = [ScalarHeads.load(RESULTS / tag) for tag in s.head_tags()]
    rows = np.arange(len(fr.shot_index))
    reading = read_many(psi, workers) if any(h.cfg.residual for h in heads) else None
    head = np.mean([h.predict(fr, rows, reading if h.cfg.residual else None) for h in heads], 0)
    t0 = time.time()
    out, _, ok = s.finisher(heads[0].y_std).many(psi, head, workers)
    logger.info(f"finished {len(out)} frames in {time.time() - t0:.0f}s, {ok.mean():.3f} accepted")
    return out
