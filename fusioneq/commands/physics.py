"""MAST transfer by physics: coil flux + a plasma current placed from Thomson, calibrated on DIII-D, and q95 / betaN
from the resulting boundary. Also the checks that score each physics piece on held-out DIII-D shots."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from sklearn.metrics import r2_score

from fusioneq.data import LABEL_COLUMNS, list_shots, read_row
from fusioneq.equilibrium import GradShafranov, Profile
from fusioneq.evaluation import score_frames, shot_rows
from fusioneq.greens import coil_basis
from fusioneq.mast_geometry import (
    FREEGS_MAST_COILS,
    FREEGS_MAST_SOLENOID,
    FREEGS_MAST_SYM_SOLENOID,
    SOLENOID,
    BasisDiff,
    SolenoidSpec,
    freegs_basis,
)
from fusioneq.physics import (
    SOLENOID_COLUMN,
    Calibration,
    Fill,
    MastCache,
    PhysicsEval,
    ScalarReplay,
    diii_d_geometry,
    ecoil_flux,
    frames_parts,
    gs_psi,
    shot_pressure,
)
from fusioneq.plasma import PlasmaShape
from fusioneq.plasma_region import MastRegion
from fusioneq.scalars import (
    SCALAR_CALIB,
    TF_COLUMN,
    PressureMode,
    ScalarCalibration,
    raw_scalars,
    tracking_stats,
)
from fusioneq.score import Scorer, summary_line
from fusioneq.settings import CACHE_ROOT, DATA_ROOT, DEVICE

FILLS: tuple[Fill, ...] = ("median", "shot")


def physics_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    fit_shots: int = 150,
    score_shots: int = 40,
    kappa: float = 1.6,
    alpha: float = 1.0,
) -> None:
    """Fit the physics Calibration on DIII-D train shots and score it on held-out ones (geometry from EFIT labels)."""
    ev = PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root)
    shape = PlasmaShape(kappa, alpha)

    def parts(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vac, plasma = frames_parts(ev.fr, rows, diii_d_geometry(ev.fr, rows, ev.shot_dir), shape, ev.G)
        return vac, plasma, ecoil_flux(ev.fr, rows, ev.coil_file)

    fit_rows = np.flatnonzero(shot_rows(ev.fr, ev.train, fit_shots))
    vac, plasma, ecoil = parts(fit_rows)
    calib = Calibration.fit(vac, ecoil, plasma, np.asarray(ev.fr.psi[fit_rows], np.float32))
    calib.save(cache_root / "phys_calib.json")
    logger.info(f"calibration on {fit_shots} train shots: {calib}")
    vac, plasma, ecoil = parts(ev.rows)
    for name, psi in [
        ("vacuum only", vac),
        ("vacuum + plasma, uncalibrated", vac + plasma),
        ("calibrated, no E-coil turns (as on MAST)", calib.apply(vac, plasma)),
        ("calibrated", calib.apply(vac, plasma, ecoil)),
    ]:
        logger.info(f"{name}: {summary_line(ev.score(psi))}")


def gs_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    iters: str = "0,10,20",
    alphas: str = "2",
    gammas: str = "1",
    beta0s: str = "0.5",
    kappa: float = 1.6,
    free: bool = False,
) -> None:
    """Score the Grad-Shafranov solve on held-out DIII-D shots (calibrated vacuum incl. E-coil turns); iters=0 is the
    plain superposition, so it reproduces physics-check's calibrated row. --free lets the current centroid move."""
    ev = PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root)
    fr, rows, calib = ev.fr, ev.rows, ev.calib
    vac = np.asarray(fr.psi_vac[rows], np.float32) + calib.ecoil_turns * ecoil_flux(fr, rows, ev.coil_file)
    geom = diii_d_geometry(fr, rows, ev.shot_dir)
    solver = GradShafranov(fr.grid_R, fr.grid_Z, Scorer("DIII-D").mask_coarse, ev.G, DEVICE)
    floats = [list(map(float, x.split(","))) for x in (alphas, gammas, beta0s)]
    for n, a, g, b in itertools.product(map(int, iters.split(",")), *floats):
        prof = Profile(a, g, b)
        raw = gs_psi(solver, vac, fr.ip[rows], geom, PlasmaShape(kappa), prof, n, calib.plasma_ratio, not free)
        s = ev.score(calib.scale * raw + calib.offset)
        cons = " ".join(f"{k} {v:.2f}" for k, v in (s.get("r2_cons_each") or {}).items())
        logger.info(f"iters={n} alpha={a} gamma={g} beta0={b} free={free}: {summary_line(s)} | {cons}")


def scalar_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    fit_shots: int = 400,
    pressure: PressureMode = "peak",
) -> None:
    """Fit q95/betaN physics constants on DIII-D train shots, report R2 on held-out shots (EFIT boundary geometry)."""
    ev = PhysicsEval.load(cache_root, holdout, data_root=data_root)
    fr = ev.fr

    def raw(rows: np.ndarray) -> np.ndarray:
        geom = diii_d_geometry(fr, rows, ev.shot_dir)
        pe = shot_pressure(fr, rows, ev.shot_dir, pressure=pressure, shape=geom[:, [0, 1, 3]], z0=geom[:, 2])
        shape = geom[:, [0, 1, 3, 4]]
        return raw_scalars(shape, fr.currents[rows, TF_COLUMN], fr.ip[rows], pe, "DIII-D")

    fit_rows = np.flatnonzero(shot_rows(fr, ev.train, fit_shots))
    calib = ScalarCalibration.fit(raw(fit_rows), np.asarray(fr.scalars[fit_rows, :2], np.float64))
    calib.save(cache_root / SCALAR_CALIB[pressure])
    rows = np.nonzero(ev.held)[0]
    pred, y = calib.apply(raw(rows)), np.asarray(fr.scalars[rows, :2], np.float64)
    ok = np.isfinite(y)
    r2 = [r2_score(y[ok[:, k], k], pred[ok[:, k], k]) for k in range(2)]
    logger.info(f"{calib}: held-out R2 q95 {r2[0]:.3f}, betaN {r2[1]:.3f}, R2_qb {np.mean(r2):.3f}")


def mast_physics(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    kappa: float = 1.6,
    alpha: float = 1.0,
    frac: float = 0.25,
    out: str = "psi_phys.npy",
    config: str = "mast_public_test",
) -> None:
    """Build `out` (calibrated vacuum + plasma ellipse) for MAST from Thomson geometry; no MAST labels used."""
    MastCache(cache_root, data_root, config).flux(PlasmaShape(kappa, alpha), frac, out)


def mast_scalars(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    workers: int = 24,
    geometry: Literal["boundary", "thomson"] = "boundary",
    eps_max: float = float("inf"),
    out: str = "scal_phys.npy",
    psi_file: str = "psi_phys.npy",
    pressure: PressureMode = "peak",
    config: str = "mast_public_test",
    gate: bool = True,
    region: MastRegion = "none",
    fill: Fill = "median",
) -> None:
    """Build MAST q95/betaN (`out` in the MAST cache) from the boundary of `psi_file`; no MAST labels used.
    geometry="thomson" takes R0 and a from the Thomson density extent (plasma_geom.npy) and keeps the boundary's
    kappa and delta. The boundary is extracted inside a geometry-only `region`, never the scorer's MAST envelope
    (rule 3); --no-gate keeps every extracted boundary instead of dropping implausible ones; `fill` takes the
    global median or the frame's own shot (fusioneq.physics.fill_from_shot) for the gaps."""
    cache = MastCache(cache_root, data_root, config)
    cache.scalars(psi_file, geometry, eps_max, out, workers, pressure, gate, region, fill)


def mast_scal_compare(
    files: str,
    ref: str = "scal_vol_eps065_a2f04.npy",
    config: str = "mast_public_test",
    cache_root: Path = CACHE_ROOT,
) -> None:
    """Label-free comparison of comma-separated MAST scalar files with `ref` (the masked file by default):
    distributions, correlation, mean |diff| and within-shot smoothness (fusioneq.scalars.tracking_stats)."""
    cache = MastCache(cache_root, config=config)
    reference = np.load(cache.dir / ref)
    for name in [ref, *files.split(",")]:
        stats = tracking_stats(np.load(cache.dir / name), reference, cache.fr.shot_index)
        logger.info(f"{name}: " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))


REGION_VARIANTS: dict[str, tuple[str, bool, Fill]] = {
    f"{region} gate={gate} fill={fill}": (region, gate, fill)
    for region, gate, fill in [
        ("envelope", False, "median"),
        ("wall", False, "median"),
        *itertools.product(("none", "wall"), (True,), FILLS),
    ]
}


def region_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    kappa: float = 1.6,
    alpha: float = 2.0,
    eps_max: float = 0.65,
    workers: int = 8,
) -> None:
    """The MAST q95 / betaN path replayed on held-out DIII-D shots, scored against their labels (allowed on DIII-D).
    The physics flux puts its ellipse at the label (R0, a) on the midplane, as MAST places it from Thomson; its LCFS
    is extracted inside the scorer's envelope (the old masked MAST analogue), nowhere, or the FreeGS DIII-D wall;
    gated against the label (R0, a) and gaps filled by the global median or the frame's own shot."""
    ev = PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root)
    replay = ScalarReplay(ev, ev.rows, eps_max, workers)
    geom = diii_d_geometry(ev.fr, replay.rows, ev.shot_dir)
    place = np.c_[geom[:, :2], np.zeros(len(geom))]
    psi = replay.physics_flux(place, PlasmaShape(kappa, alpha))
    label = replay.predict(geom[:, [0, 1, 3, 4]])
    logger.info(f"label boundary: R2 q95 / betaN {replay.r2(label)}")
    for name, (pred, gaps) in replay.extraction(psi, place, REGION_VARIANTS).items():
        stats = tracking_stats(pred, label, replay.shot_index)
        logger.info(
            f"{name}: filled {gaps:.1%}, R2 q95 / betaN {replay.r2(pred)}, "
            f"corr with label-boundary q95 {stats['q95_corr']:.3f} betaN {stats['betaN_corr']:.3f}"
        )


def mast_offset(
    psi_file: str = "psi_phys.npy",
    out: str = "psi_phys_offset.npy",
    config: str = "mast_public_test",
    fit_config: str = "mast_demo",
    cache_root: Path = CACHE_ROOT,
) -> None:
    """Save `out` in `config`: `psi_file` plus a per-frame offset a + b * I_sol fitted on the labelled `fit_config`
    (the same psi_file built there by mast-physics --config mast_demo). This uses MAST labels: the demo shots."""
    labelled = MastCache(cache_root, config=fit_config)
    MastCache(cache_root, config=config).offset_corrected(psi_file, out, labelled)


def mast_scalar_fit(
    scal_file: str = "scal_phys.npy",
    out: str = "scal_phys_fit.npy",
    config: str = "mast_public_test",
    fit_config: str = "mast_demo",
    cache_root: Path = CACHE_ROOT,
) -> None:
    """Save `out` in `config`: `scal_file` mapped per scalar by a line fitted on the labelled `fit_config` (the same
    scal_file built there by mast-scalars --config mast_demo). This uses MAST labels: the demo shots."""
    labelled = MastCache(cache_root, config=fit_config)
    MastCache(cache_root, config=config).scalars_corrected(scal_file, out, labelled)


def mast_check(
    psi_file: str = "psi_phys.npy",
    scal_file: str = "scal_phys.npy",
    config: str = "mast_demo",
    cache_root: Path = CACHE_ROOT,
) -> None:
    """Score `psi_file` / `scal_file` from a MAST cache with labels (the starter's three demo shots) by the official
    MAST scorer. Build them first with mast-physics / mast-scalars --config mast_demo. Validation only: three shots
    set R2 against their own variance, so compare candidates here rather than read the level as the official one."""
    cache = MastCache(cache_root, config=config)
    res = score_frames(cache.fr, *cache.prediction(psi_file, scal_file), machine="MAST", cache_dir=cache_root / "refs")
    logger.info(f"{config} {psi_file} + {scal_file}: {summary_line(res)} r2_each={res['r2_each']}")


def mast_geometry(cache_root: Path = CACHE_ROOT, data_root: Path = DATA_ROOT, config: str = "mast_public_test") -> None:
    """Shipped MAST coil flux vs FreeGS `MAST()` per unit current (geometry and test inputs only, no labels)."""
    row = read_row(list_shots(data_root, config)[0], exclude=LABEL_COLUMNS)
    grid_R, grid_Z = np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)
    coils = {k: np.asarray(row[k]) for k in row if k.startswith("coil_")}
    columns = [SOLENOID, *FREEGS_MAST_COILS]
    ours = coil_basis(grid_R, grid_Z, coils, columns).astype(np.float64)
    for col, diff in zip(columns, map(BasisDiff.of, ours, freegs_basis(grid_R, grid_Z, columns)), strict=True):
        logger.info(f"{col}: {diff}")
    fr = MastCache(cache_root, data_root, config).fr
    i_sol = fr.currents[:, SOLENOID_COLUMN].astype(np.float64)
    scale = abs(Calibration.from_cache(cache_root).scale)
    alternatives = {
        "FreeGS MAST()": FREEGS_MAST_SOLENOID,
        "FreeGS MAST_sym()": FREEGS_MAST_SYM_SOLENOID,
        "FreeGS radius/height, shipped turns": SolenoidSpec(
            0.15, -1.4, 1.4, int((coils["coil_input_column"] == SOLENOID).sum())
        ),
    }
    for name, spec in alternatives.items():
        diff = BasisDiff.of(ours[0], spec.flux(grid_R, grid_Z))
        level = scale * diff.level_diff * i_sol
        logger.info(f"solenoid vs {name}: {diff}; per-frame level shift std {level.std():.4f} Wb/rad")


COMMANDS = [
    physics_check,
    gs_check,
    scalar_check,
    mast_physics,
    mast_scalars,
    mast_scal_compare,
    region_check,
    mast_offset,
    mast_scalar_fit,
    mast_check,
    mast_geometry,
]
