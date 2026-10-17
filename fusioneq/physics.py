"""Physics-only flux (vacuum + plasma ellipse) for machines the network cannot ingest, and its DIII-D calibration.

DIII-D: geometry from the EFIT boundary labels (allowed) fits the Calibration and bounds what this model can score.
MAST: geometry from Thomson density only; no MAST equilibrium data is read.

The DIII-D E-coil ships as one rectangle carrying per-turn current, so its basis counts a single turn and the
calibration fits the missing turns. The flux of a tall solenoid is nearly uniform over the grid, so without them
every frame is off by a current-dependent constant. MAST ships its solenoid turn by turn and needs no such term.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pyarrow.parquet as pq
import torch
from loguru import logger

from fusioneq.cache import Frames, shot_bounds
from fusioneq.data import LABEL_COLUMNS, MAST_COILS, list_shots, profile_on_efit, read_row, thomson_geometry
from fusioneq.equilibrium import GradShafranov, Profile
from fusioneq.evaluation import HeldOut
from fusioneq.greens import cached_grid_greens, coil_basis
from fusioneq.mast_shape import KAPPA_BAND
from fusioneq.plasma import PlasmaShape, current_density, density_extent
from fusioneq.plasma_region import MastRegion, Region
from fusioneq.scalars import (
    SCALAR_CALIB,
    TF_COLUMN,
    PressureMode,
    ScalarCalibration,
    peak_pressure,
    raw_scalars,
    volume_pressure,
)
from fusioneq.score import AXIS_SIGN, Scorer, boundary_shape, lcfs_shapes_parallel
from fusioneq.settings import CACHE_ROOT, DATA_ROOT
from fusioneq.shape_map import r2

CHUNK = 4096
# our calibrated MAST flux puts the axis at a minimum; the scorer flips it (sign -1 on the public test)
MAST_AXIS_SIGN = -1.0
Array = TypeVar("Array", np.ndarray, torch.Tensor)
ECOIL = "magnetics_ECOILA"
ECOIL_COLUMN = 0  # data.D3D_COILS[0]
SOLENOID_COLUMN = MAST_COILS.index("sol_current")
Fill = Literal["median", "shot"]


@dataclass(frozen=True)
class Calibration:
    """psi = scale * (vac + ecoil_turns * ecoil + plasma_ratio * plasma) + offset, fitted on DIII-D train shots."""

    scale: float
    plasma_ratio: float
    offset: float
    ecoil_turns: float

    @classmethod
    def fit(
        cls, vac: np.ndarray, ecoil: np.ndarray, plasma: np.ndarray, psi: np.ndarray, stride: int = 13
    ) -> Calibration:
        # subsample before stacking: the full (N*4225, 4) float64 design matrix runs to gigabytes
        cols = [a.ravel()[::stride] for a in (vac, ecoil, plasma)]
        X = np.stack([*cols, np.ones_like(cols[0])], 1)
        w, *_ = np.linalg.lstsq(X, psi.ravel()[::stride], rcond=None)
        return cls(float(w[0]), float(w[2] / w[0]), float(w[3]), float(w[1] / w[0]))

    def apply(self, vac: Array, plasma: Array, ecoil: Array | float = 0.0) -> Array:
        """Works on numpy (returned as float32) and on torch tensors (returned as is)."""
        out = self.scale * (vac + self.ecoil_turns * ecoil + self.plasma_ratio * plasma) + self.offset
        return out.astype(np.float32) if isinstance(out, np.ndarray) else out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_cache(cls, cache_root: Path) -> Calibration:
        """The calibration `physics-check` fits and saves as phys_calib.json."""
        return cls.load(cache_root / "phys_calib.json")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class SolenoidOffset:
    """Per-frame psi offset a + b * I_sol (solenoid current, kA) in the scorer's sign: the flux of vacuum-vessel
    currents, which the inputs omit (README) and the solenoid swing drives. Fitted on the starter's three labelled
    MAST demo shots, so it is the one MAST-label-fitted term in the pipeline."""

    a: float
    b: float

    @classmethod
    def fit(cls, psi: np.ndarray, truth: np.ndarray, i_sol: np.ndarray) -> SolenoidOffset:
        off = truth.mean((1, 2), dtype=np.float64) - MAST_AXIS_SIGN * psi.mean((1, 2), dtype=np.float64)
        (a, b), *_ = np.linalg.lstsq(np.c_[np.ones(len(off)), i_sol], off, rcond=None)
        return cls(float(a), float(b))

    def apply(self, psi: np.ndarray, i_sol: np.ndarray) -> np.ndarray:
        off = (MAST_AXIS_SIGN * (self.a + self.b * np.asarray(i_sol, np.float64))).astype(np.float32)
        return np.asarray(psi, np.float32) + off[:, None, None]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> SolenoidOffset:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class DemoScalarFit:
    """Per-scalar line truth = a + b * prediction for q95 and betaN, fitted on the starter's labelled MAST demo shots:
    our q95 sits ~1 low and our betaN spread is too wide. Like SolenoidOffset, it is fitted on MAST labels."""

    q_a: float
    q_b: float
    bn_a: float
    bn_b: float

    @classmethod
    def fit(cls, pred: np.ndarray, truth: np.ndarray) -> DemoScalarFit:
        lines = []
        for j in range(2):
            ok = np.isfinite(truth[:, j]) & np.isfinite(pred[:, j])
            b, a = np.polyfit(np.asarray(pred[ok, j], np.float64), np.asarray(truth[ok, j], np.float64), 1)
            lines += [float(a), float(b)]
        return cls(*lines)

    def apply(self, pred: np.ndarray) -> np.ndarray:
        a = np.array([self.q_a, self.bn_a], np.float32)
        return a + np.array([self.q_b, self.bn_b], np.float32) * np.asarray(pred, np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> DemoScalarFit:
        return cls(**json.loads(path.read_text()))


def ellipse_weights(grid_R: np.ndarray, grid_Z: np.ndarray, geom: np.ndarray, shape: PlasmaShape) -> np.ndarray:
    """(N, nz * nr) ellipse current weights (each row sums to 1) for geom rows (R0, a, Z0)."""
    return np.stack([current_density(grid_R, grid_Z, r0, a, z0, shape).ravel() for r0, a, z0 in geom[:, :3]])


def plasma_psi(
    G: np.ndarray, grid_R: np.ndarray, grid_Z: np.ndarray, ip_ka: np.ndarray, geom: np.ndarray, shape: PlasmaShape
) -> np.ndarray:
    """(N, 65, 65) plasma flux for N frames; geom rows are (R0, a, Z0)."""
    out = np.empty((len(ip_ka), len(grid_Z), len(grid_R)), np.float32)
    for i in range(0, len(ip_ka), CHUNK):
        J = ellipse_weights(grid_R, grid_Z, geom[i : i + CHUNK], shape) * (ip_ka[i : i + CHUNK, None] * 1e3)
        out[i : i + CHUNK] = (J.astype(np.float32) @ G.T).reshape(-1, len(grid_Z), len(grid_R))
    return out


def ecoil_basis(grid_R: np.ndarray, grid_Z: np.ndarray, coil_file: Path) -> np.ndarray:
    """(65, 65) single-turn E-coil flux per kA; the DIII-D coil table is identical across shots."""
    coils = pq.read_table(
        coil_file, columns=["coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height"]
    ).to_pylist()[0]
    return coil_basis(grid_R, grid_Z, coils, [ECOIL])[0]


def ecoil_flux(fr: Frames, rows: np.ndarray, coil_file: Path) -> np.ndarray:
    """(N, 65, 65) single-turn E-coil flux per frame."""
    return np.asarray(fr.currents[rows, ECOIL_COLUMN], np.float32)[:, None, None] * ecoil_basis(
        fr.grid_R, fr.grid_Z, coil_file
    )


def lcfs_geometry(path: Path) -> np.ndarray:
    """(T, 5) R0, a, Z0, kappa, delta per EFIT frame from a DIII-D shot's labelled boundary (NaN where missing)."""
    t = pq.read_table(path, columns=["efit_lcfs_r", "efit_lcfs_z", "efit_lcfs_n"]).to_pylist()[0]
    rows = []
    for r, z, n in zip(t["efit_lcfs_r"], t["efit_lcfs_z"], t["efit_lcfs_n"], strict=True):
        k = int(n) if n and np.isfinite(n) else 0
        if k < 3:
            rows.append((np.nan,) * 5)
            continue
        r, z = np.asarray(r[:k], float), np.asarray(z[:k], float)
        r0, a, kappa, delta = boundary_shape(r, z)
        rows.append((r0, a, (z.max() + z.min()) / 2, kappa, delta))
    return np.asarray(rows, np.float32)


def thomson_frame_geometry(path: Path, frac: float) -> np.ndarray:
    """(T, 3) R0, a, Z0=0 per EFIT frame from a MAST shot's Thomson density, gaps filled from the shot's median."""
    row = read_row(path, exclude=LABEL_COLUMNS)
    efit_t = np.asarray(row["efit_times"], float)
    R, _, ne, _ = thomson_geometry(row)
    prof = profile_on_efit(np.asarray(row["thomson_core_times"], float), ne, efit_t)
    geom = np.full((len(efit_t), 3), np.nan, np.float32)
    for k, n in enumerate(prof):
        ext = density_extent(R, n, frac)
        if ext is not None and ext[1] - ext[0] > 0.1:
            geom[k] = ((ext[0] + ext[1]) / 2, (ext[1] - ext[0]) / 2, 0.0)
    return geom


def plausible_boundary(shape: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """shape rows (R0, a, kappa, delta) of our extracted LCFS, NaN where it cannot be this frame's plasma: kappa
    outside KAPPA_BAND, or a centre or minor radius the Thomson extent (geom rows R0, a) rules out. Without the
    scorer's envelope (rule 3) the extractor can close around a coil or the field null instead."""
    r0, a = geom[:, 0], geom[:, 1]
    ok = (
        (shape[:, 2] >= KAPPA_BAND[0])
        & (shape[:, 2] <= KAPPA_BAND[1])
        & (np.abs(shape[:, 0] - r0) <= 0.5 * a)
        & (shape[:, 1] >= 0.5 * a)
        & (shape[:, 1] <= 1.5 * a)
    )
    return np.where(ok[:, None], shape, np.nan)


def fill_gaps(geom: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    ok = np.isfinite(geom).all(1)
    fill = np.nanmedian(geom[ok], 0) if ok.any() else fallback
    geom = geom.copy()
    geom[~ok] = fill
    return geom


def fill_from_shot(shape: np.ndarray, shot_index: np.ndarray, thomson: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(filled shape, source) for shape rows (R0, a, kappa, delta) with NaN rows: each gap is interpolated over the
    frame index from the plausible frames of its own shot (held at the shot's ends); a shot with none takes its own
    Thomson (R0, a) with the median plausible (kappa, delta). source: 0 own boundary, 1 same shot, 2 Thomson."""
    ok = np.isfinite(shape).all(1)
    out, source = shape.copy(), np.where(ok, 0, 1).astype(np.int8)
    shape_median = np.nanmedian(shape[ok, 2:], 0) if ok.any() else np.array([1.6, 0.0])
    for s, e in zip(*shot_bounds(shot_index), strict=True):
        good, idx = ok[s:e], np.arange(e - s)
        if good.any():
            for j in range(shape.shape[1]):
                out[s:e, j] = np.interp(idx, idx[good], shape[s:e, j][good])
        else:
            out[s:e] = np.c_[thomson[s:e, :2], np.broadcast_to(shape_median, (e - s, 2))]
            source[s:e] = 2
    return out, source


def fill_shapes(shape: np.ndarray, fill: Fill, shot_index: np.ndarray, thomson: np.ndarray) -> np.ndarray:
    """Every NaN row of `shape` filled by the global median (`median`) or from its own shot (`shot`, fill_from_shot)."""
    if fill == "median":
        return fill_gaps(shape, np.nanmedian(shape, 0))
    filled, source = fill_from_shot(shape, shot_index, thomson)
    logger.info(f"fill sources own / same shot / Thomson: {np.bincount(source, minlength=3) / len(source)}")
    return filled


def mast_geometry(data_root: Path, config: str, frac: float, workers: int = 12) -> np.ndarray:
    paths = list_shots(data_root, config)
    with ProcessPoolExecutor(workers) as ex:
        per_shot = list(ex.map(thomson_frame_geometry, paths, [frac] * len(paths), chunksize=8))
    pooled = np.concatenate(per_shot)
    fallback = np.nanmedian(pooled[np.isfinite(pooled).all(1)], 0)
    return np.concatenate([fill_gaps(g, fallback) for g in per_shot])


def diii_d_geometry(fr: Frames, rows: np.ndarray, shot_dir: Path) -> np.ndarray:
    """(N, 5) R0, a, Z0, kappa, delta for DIII-D frames `rows` (whole shots) from the labelled boundary, gaps filled
    from the median."""
    geom = np.concatenate([lcfs_geometry(shot_dir / fr.shots[i]["file"]) for i in np.unique(fr.shot_index[rows])])
    return np.where(np.isfinite(geom), geom, np.nanmedian(geom, 0))


def frames_parts(
    fr: Frames, rows: np.ndarray, geom: np.ndarray, shape: PlasmaShape, G: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(vacuum, plasma) flux for frames `rows`, each (N, 65, 65)."""
    plasma = plasma_psi(G, fr.grid_R, fr.grid_Z, fr.ip[rows].astype(np.float64), geom, shape)
    return fr.psi_vac[rows].astype(np.float32), plasma


def gs_psi(
    solver: GradShafranov,
    vac: np.ndarray,
    ip_ka: np.ndarray,
    geom: np.ndarray,
    shape: PlasmaShape,
    profile: Profile,
    iters: int,
    ip_scale: float = 1.0,
    pin: bool = True,
) -> np.ndarray:
    """(N, 65, 65) raw-orientation flux from the Grad-Shafranov solve, started from the ellipse current of `geom`.
    iters = 0 is the plain superposition vac + G J_ellipse."""
    out = np.empty(vac.shape, np.float32)
    grid_R, grid_Z = solver.grid_R, solver.grid_Z
    for i in range(0, len(vac), CHUNK):
        sl = slice(i, i + CHUNK)
        j0 = ellipse_weights(grid_R, grid_Z, geom[sl], shape)
        ip_a = np.asarray(ip_ka[sl], np.float64) * 1e3 * ip_scale
        psi, _ = solver.solve(vac[sl].reshape(len(j0), -1), ip_a, j0, profile, iters, pin)
        out[sl] = psi.reshape(-1, *vac.shape[1:])
    return out


@dataclass
class PhysicsEval(HeldOut):
    """The held-out DIII-D shots plus the physics inputs the checks share: raw shot files, calibration, Green's."""

    data_root: Path = DATA_ROOT

    @property
    def shot_dir(self) -> Path:
        return self.data_root / "data" / "diii_d_train"

    @property
    def coil_file(self) -> Path:
        return self.shot_dir / self.fr.shots[0]["file"]

    @cached_property
    def calib(self) -> Calibration:
        return Calibration.from_cache(self.cache_root)

    @cached_property
    def G(self) -> np.ndarray:
        return cached_grid_greens(self.cache_root / "diii_d_train", self.fr.grid_R, self.fr.grid_Z)


def shot_pressure(
    fr: Frames,
    rows: np.ndarray,
    shot_dir: Path,
    workers: int = 12,
    pressure: PressureMode = "peak",
    shape: np.ndarray | None = None,
    z0: np.ndarray | None = None,
) -> np.ndarray:
    """(N,) Thomson electron pressure for frames `rows` (whole shots in cache order), peak or volume-averaged."""
    shots, counts = np.unique(fr.shot_index[rows], return_counts=True)
    files = [shot_dir / fr.shots[i]["file"] for i in shots]
    with ProcessPoolExecutor(workers) as ex:
        if pressure == "peak":
            return np.concatenate(list(ex.map(peak_pressure, files, chunksize=8)))
        cut = np.cumsum(counts)[:-1]
        z0s = [None] * len(files) if z0 is None else np.split(z0, cut)
        return np.concatenate(list(ex.map(volume_pressure, files, np.split(shape, cut), z0s, chunksize=8)))


def perturb_placement(
    geom: np.ndarray,
    shot_index: np.ndarray,
    seed: int,
    shot_sd: tuple[float, float] = (0.02, 0.08),
    frame_sd: float = 0.04,
) -> np.ndarray:
    """geom rows (R0, a, ...) with a Thomson-like placement error: per shot an R0 shift (shot_sd[0], m) and a relative a
    bias (shot_sd[1]), and per frame a relative a jitter (frame_sd). A DIII-D stand-in for MAST's Thomson (R0, a),
    whose error against the LCFS no label-free check can measure."""
    rng = np.random.default_rng(seed)
    inv = np.unique(shot_index, return_inverse=True)[1]
    dr, da = (rng.normal(0.0, sd, inv.max() + 1)[inv] for sd in shot_sd)
    out = np.array(geom, np.float64)
    out[:, 0] += dr
    out[:, 1] *= (1 + da) * (1 + rng.normal(0.0, frame_sd, len(out)))
    return out


EXTRACTION_BASELINES: dict[str, tuple[str, bool, Fill]] = {  # name: (region, gate, fill) of region-check's variants
    "envelope, ungated (masked MAST analogue, 0.320)": ("envelope", False, "median"),
    "no mask, gated, median fill (22d99887, 0.208)": ("none", True, "median"),
    "FreeGS wall, gated, shot fill (1b4990da, 0.115)": ("wall", True, "shot"),
}


@dataclass
class ScalarReplay:
    """The MAST q95 / betaN formula and DIII-D calibration on DIII-D frames `rows` (whole shots), scored against their
    labels: the replay that checks a MAST scalar procedure where labels are allowed."""

    ev: PhysicsEval
    rows: np.ndarray
    eps_max: float = 0.65
    workers: int = 8

    @cached_property
    def calib(self) -> ScalarCalibration:
        return ScalarCalibration.load(self.ev.cache_root / SCALAR_CALIB["volume"])

    @property
    def truth(self) -> np.ndarray:
        return np.asarray(self.ev.fr.scalars[self.rows, :2], np.float64)

    @property
    def shot_index(self) -> np.ndarray:
        return self.ev.fr.shot_index[self.rows]

    def predict(self, shape: np.ndarray, z0: np.ndarray | None = None) -> np.ndarray:
        """(N, 2) q95 / betaN from shape rows (R0, a, kappa, delta), volume pressure on ellipses centred at z0."""
        fr, r = self.ev.fr, self.rows
        pe = shot_pressure(fr, r, self.ev.shot_dir, self.workers, "volume", shape[:, :3], z0)
        return self.calib.apply(raw_scalars(shape, fr.currents[r, TF_COLUMN], fr.ip[r], pe, "DIII-D", self.eps_max))

    def r2(self, pred: np.ndarray) -> tuple[float, float]:
        """Pooled R2 of q95 and betaN over frames with a label (the scorer's per-scalar form)."""
        return r2(self.truth[:, 0], pred[:, 0]), r2(self.truth[:, 1], pred[:, 1])

    def physics_flux(self, place: np.ndarray, shape: PlasmaShape) -> np.ndarray:
        """(N, 65, 65) the calibrated physics flux with its plasma ellipse at place rows (R0, a, Z0)."""
        fr, ev = self.ev.fr, self.ev
        vac, plasma = frames_parts(fr, self.rows, place, shape, ev.G)
        return ev.calib.apply(vac, plasma, ecoil_flux(fr, self.rows, ev.coil_file))

    def extraction(
        self, psi: np.ndarray, place: np.ndarray, variants: dict[str, tuple[str, bool, Fill]] = EXTRACTION_BASELINES
    ) -> dict[str, tuple[np.ndarray, float]]:
        """{name: ((N, 2) scalars, share of frames filled)} per variant (region, gate, fill) on flux `psi`: the LCFS
        extracted inside the region (diii_d_regions), gated against place rows (R0, a) where asked, gaps filled."""
        fr, out = self.ev.fr, {}
        regions = {v[0] for v in variants.values()}
        masks = {k: m for k, m in diii_d_regions(fr.grid_R, fr.grid_Z).items() if k in regions}
        shapes = {
            k: lcfs_shapes_parallel(psi, fr.grid_R, fr.grid_Z, "DIII-D", AXIS_SIGN["DIII-D"], m, self.workers)
            for k, m in masks.items()
        }
        for name, (region, gate, fill) in variants.items():
            shape = plausible_boundary(shapes[region], place) if gate else shapes[region]
            gaps = float(np.isnan(shape).any(1).mean())
            out[name] = (self.predict(fill_shapes(shape, fill, self.shot_index, place)), gaps)
        return out


def diii_d_regions(grid_R: np.ndarray, grid_Z: np.ndarray) -> dict[str, np.ndarray | None]:
    """DIII-D LCFS extraction regions: the scorer's envelope (the old masked MAST analogue), none, the FreeGS wall."""
    return {"envelope": Scorer("DIII-D").mask_coarse, "none": None, "wall": Region.diii_d_wall(grid_R, grid_Z).mask}


@dataclass
class MastCache:
    """A MAST config's cache and the physics predictions written into it; no MAST labels are used. `mast_demo` (the
    starter's three labelled demo shots) runs the same pipeline so `mast-check` can score it."""

    cache_root: Path = CACHE_ROOT
    data_root: Path = DATA_ROOT
    config: str = "mast_public_test"

    @property
    def dir(self) -> Path:
        return self.cache_root / self.config

    @cached_property
    def fr(self) -> Frames:
        return Frames.load(self.dir)

    @cached_property
    def calib(self) -> Calibration:
        return Calibration.from_cache(self.cache_root)

    def prediction(self, psi_file: str, scal_file: str) -> tuple[np.ndarray, np.ndarray]:
        """(psi, scalars) built by mast-physics / mast-scalars, as float32."""
        missing = [n for n in (psi_file, scal_file) if not (self.dir / n).exists()]
        if missing:
            raise FileNotFoundError(f"{self.config}: missing {missing}; run mast-physics / mast-scalars first")
        return np.load(self.dir / psi_file).astype(np.float32), np.load(self.dir / scal_file).astype(np.float32)

    def offset_corrected(self, psi_file: str, out: str, labelled: MastCache) -> SolenoidOffset:
        """Save `out`: `psi_file` plus the SolenoidOffset fitted on the same file in `labelled` (a cache with psi);
        the fitted numbers go to mast_offset.json in the cache root, beside phys_calib.json."""
        fit = SolenoidOffset.fit(
            np.load(labelled.dir / psi_file), labelled.fr.psi, labelled.fr.currents[:, SOLENOID_COLUMN]
        )
        fit.save(self.cache_root / "mast_offset.json")
        psi = fit.apply(np.load(self.dir / psi_file), self.fr.currents[:, SOLENOID_COLUMN])
        np.save(self.dir / out, psi.astype(np.float16))
        logger.info(f"{self.config} {out}: {psi_file} + {fit} fitted on {labelled.config}")
        return fit

    def scalars_corrected(self, scal_file: str, out: str, labelled: MastCache) -> DemoScalarFit:
        """Save `out`: `scal_file` mapped by the DemoScalarFit fitted on the same file in `labelled`; the fitted
        lines go to mast_scalar_fit.json in the cache root."""
        fit = DemoScalarFit.fit(np.load(labelled.dir / scal_file), np.asarray(labelled.fr.scalars[:, :2]))
        fit.save(self.cache_root / "mast_scalar_fit.json")
        np.save(self.dir / out, fit.apply(np.load(self.dir / scal_file)))
        logger.info(f"{self.config} {out}: {scal_file} mapped by {fit} fitted on {labelled.config}")
        return fit

    def flux(self, shape: PlasmaShape, frac: float, out: str) -> np.ndarray:
        """Calibrated vacuum + a plasma ellipse placed from the Thomson density extent, saved as `out`."""
        calib = self.calib
        geom = mast_geometry(self.data_root, self.config, frac)
        if len(geom) != len(self.fr.shot_index):
            raise ValueError(f"geometry has {len(geom)} frames, the cache {len(self.fr.shot_index)}")
        G = cached_grid_greens(self.dir, self.fr.grid_R, self.fr.grid_Z)
        psi = calib.apply(*frames_parts(self.fr, np.arange(len(geom)), geom, shape, G))
        np.save(self.dir / out, psi.astype(np.float16))
        np.save(self.dir / "plasma_geom.npy", geom)
        ptp = np.median(np.ptp(psi.reshape(len(psi), -1), 1))
        logger.info(
            f"MAST {out} {psi.shape} with {calib}: R0 median {np.median(geom[:, 0]):.3f}, "
            f"a median {np.median(geom[:, 1]):.3f}, frame peak-to-peak median {ptp:.3f}"
        )
        return psi

    @cached_property
    def coils(self) -> dict[str, np.ndarray]:
        """The shipped coil geometry (coil_R, coil_Z, coil_width, coil_height, ...) of the config's first shot."""
        row = read_row(list_shots(self.data_root, self.config)[0], exclude=LABEL_COLUMNS)
        return {k: np.asarray(row[k]) for k in row if k.startswith("coil_")}

    def boundary_shapes(self, psi_file: str, workers: int, region: MastRegion = "none") -> np.ndarray:
        """(T, 4) R0, a, kappa, delta of the LCFS the official extractor finds in every frame of `psi_file` inside a
        geometry-only Region (none, box or chamber), never the scorer's MAST envelope (rule 3). Cached as
        lcfs_<psi stem>_<region>.npy beside the flux."""
        path = self.dir / f"lcfs_{Path(psi_file).stem}_{region}.npy"
        if path.exists():
            logger.info(f"MAST boundaries from {path.name}")
            return np.load(path)
        psi = np.load(self.dir / psi_file, mmap_mode="r")
        R, Z = np.load(self.dir / "grid_R.npy"), np.load(self.dir / "grid_Z.npy")
        mask = Region.mast(region, R, Z, self.coils if region == "chamber" else None).mask
        shapes = lcfs_shapes_parallel(psi, R, Z, "MAST", MAST_AXIS_SIGN, mask, workers)
        np.save(path, shapes)
        return shapes

    def plausible_shapes(self, psi_file: str, workers: int, region: MastRegion, gate: bool, fill: Fill) -> np.ndarray:
        """(T, 4) boundary shapes with every gap filled: implausible boundaries (plausible_boundary, if `gate`) and
        frames with none take the global median (`median`) or their own shot's neighbours (`shot`, fill_from_shot)."""
        shape = self.boundary_shapes(psi_file, workers, region)
        thomson = np.load(self.dir / "plasma_geom.npy")
        if gate:
            shape = plausible_boundary(shape, thomson)
        logger.info(f"MAST {psi_file} [{region}]: {np.isnan(shape).any(1).mean():.1%} of boundaries missing or dropped")
        return fill_shapes(shape, fill, self.fr.shot_index, thomson)

    def scalars(
        self,
        psi_file: str,
        geometry: Literal["boundary", "thomson"],
        eps_max: float,
        out: str,
        workers: int,
        pressure: PressureMode = "peak",
        gate: bool = True,
        region: MastRegion = "none",
        fill: Fill = "median",
    ) -> np.ndarray:
        """q95 / betaN from the boundary of `psi_file` (or R0, a from the Thomson extent), saved as `out`, with
        the peak or volume-averaged Thomson pressure for betaN; see plausible_shapes for `region`, `gate`, `fill`."""
        shape = self.plausible_shapes(psi_file, workers, region, gate, fill)
        if geometry == "thomson":
            shape[:, :2] = np.load(self.dir / "plasma_geom.npy")[:, :2]
        return self.shape_scalars(shape, eps_max, out, workers, pressure)

    def shape_scalars(
        self,
        shape: np.ndarray,
        eps_max: float,
        out: str,
        workers: int,
        pressure: PressureMode,
        z0: np.ndarray | None = None,
    ) -> np.ndarray:
        """q95 / betaN from shape rows (R0, a, kappa, delta) with the Thomson pressure averaged over ellipses centred
        at z0 (the midplane by default), saved as `out`."""
        calib = ScalarCalibration.load(self.cache_root / SCALAR_CALIB[pressure])
        mast = self.data_root / "data" / self.config
        pe = shot_pressure(self.fr, np.arange(len(shape)), mast, workers, pressure, shape[:, :3], z0)
        scal = calib.apply(raw_scalars(shape, self.fr.currents[:, TF_COLUMN], self.fr.ip, pe, "MAST", eps_max))
        np.save(self.dir / out, scal)
        q = np.percentile(scal, [10, 50, 90], axis=0).round(2)
        capped = np.mean(scal[:, 0] >= calib.q95_max - 1e-3)
        logger.info(
            f"MAST {out} {scal.shape}: q95 p10/50/90 {q[:, 0]} ({capped:.1%} capped), betaN p10/50/90 {q[:, 1]}"
        )
        return scal
