"""Lever 1b: per-frame plasma shape from the vacuum field, fitted on DIII-D and applied to MAST without MAST labels.

shape-fit    fit the ShapeMap on DIII-D train shots, report held-out R2 per target (DIII-D labels only)
shape-check  the whole MAST procedure replayed on the 40 held-out DIII-D scoring shots, scored against their labels
mast-shape   build the MAST candidate flux in the MAST cache (no MAST labels, no envelope mask)
derive-check label-free table of the scorer's derived scalars on a MAST flux file (no labels, no envelope mask)
geom-scal-check   q95 / betaN from placement + ShapeMap (no boundary extraction) replayed on held-out DIII-D shots
mast-geom-scalars build those MAST q95 / betaN (Thomson placement + ShapeMap; no MAST labels, no extraction)
shift-fit    fit the Shafranov-shift model of the magnetic axis on DIII-D labels (shape-check / mast-shape --shift-axis)
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from fusioneq.cache import Frames, shot_bounds
from fusioneq.data import coil_columns, list_shots, read_row
from fusioneq.evaluation import score_frames
from fusioneq.greens import cached_grid_greens, coil_basis
from fusioneq.mast_shape import (
    DERIVED,
    KAPPA_LADDER,
    SERVED,
    Deriver,
    PlasmaFlux,
    ShapeLadder,
    coil_guard,
    summary,
)
from fusioneq.physics import (
    MAST_AXIS_SIGN,
    SOLENOID_COLUMN,
    Calibration,
    MastCache,
    PhysicsEval,
    ScalarReplay,
    diii_d_geometry,
    ecoil_flux,
    lcfs_geometry,
    perturb_placement,
    plasma_psi,
    shot_pressure,
)
from fusioneq.plasma import PlasmaShape
from fusioneq.scalars import shafranov_drive
from fusioneq.score import AXIS_SIGN, Scorer, summary_line
from fusioneq.settings import CACHE_ROOT, DATA_ROOT
from fusioneq.shape_map import (
    CONS_INDEX,
    FEATURES,
    TARGETS,
    ShapeMap,
    ShiftModel,
    features,
    placement_shape,
    r2,
)

MAP_FILE = "shape_map.json"
BLOCK = 2048
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FIT_SHOTS = 800
SHIFT_FILE = "shift_model.json"
ReplayShots = Literal["scoring", "others"]


def coils_of(path: Path) -> dict[str, np.ndarray]:
    row = read_row(path)
    return {**{k: np.asarray(row[k]) for k in row if k.startswith("coil_")}, "columns": np.array(coil_columns(row))}


@dataclass
class DiiidShape:
    """DIII-D frames for the shape map: label geometry (R0, a, Z0 from the EFIT boundary), raw vacuum flux with the
    missing E-coil turns, and the scorer-derived targets in cons_targets.npz."""

    cache_root: Path = CACHE_ROOT
    data_root: Path = DATA_ROOT
    workers: int = 8

    @cached_property
    def fr(self) -> Frames:
        return Frames.load(self.cache_root / "diii_d_train")

    @cached_property
    def targets(self) -> np.lib.npyio.NpzFile:
        return np.load(self.cache_root / "diii_d_train" / "cons_targets.npz")

    @cached_property
    def calib(self) -> Calibration:
        return Calibration.from_cache(self.cache_root)

    @property
    def shot_dir(self) -> Path:
        return self.data_root / "data" / "diii_d_train"

    def geometry(self, rows: np.ndarray) -> np.ndarray:
        """(N, 3) R0, a, Z0 of the labelled boundary per frame, NaN where missing."""
        starts, _ = shot_bounds(np.asarray(self.fr.shot_index))
        sids = np.asarray(self.fr.shot_index[rows])
        uniq = np.unique(sids)
        with ProcessPoolExecutor(self.workers) as ex:
            per = dict(
                zip(uniq, ex.map(lcfs_geometry, [self.shot_dir / self.fr.shots[i]["file"] for i in uniq]), strict=True)
            )
        return np.stack([per[s][r - starts[s], :3] for s, r in zip(sids, rows, strict=True)]).astype(np.float64)

    def vacuum(self, rows: np.ndarray) -> np.ndarray:
        coil_file = self.shot_dir / self.fr.shots[0]["file"]
        vac = np.asarray(self.fr.psi_vac[rows], np.float32)
        return vac + self.calib.ecoil_turns * ecoil_flux(self.fr, rows, coil_file)

    def fit_shot_ids(self, fit_shots: int = FIT_SHOTS) -> np.ndarray:
        """The train shots shape-fit draws to fit the map on (a fixed draw)."""
        sid = np.asarray(self.fr.shot_index)
        return np.random.default_rng(0).choice(np.unique(sid[self.targets["train_rows"]]), fit_shots, replace=False)

    def features(self, rows: np.ndarray, geom: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                features(self.vacuum(r), self.fr.grid_R, self.fr.grid_Z, geom[i, 0], geom[i, 1], self.fr.ip[r])
                for i, r in blocks(rows)
            ]
        )


def blocks(rows: np.ndarray, size: int = BLOCK) -> list[tuple[np.ndarray, np.ndarray]]:
    """(positions, rows) per block of `size`."""
    return [(np.arange(s, min(s + size, len(rows))), rows[s : s + size]) for s in range(0, len(rows), size)]


def target_matrix(y: np.ndarray) -> np.ndarray:
    return y[:, [CONS_INDEX[t] for t in TARGETS]]


def group_cv(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lam: float, folds: int = 3) -> float:
    """Mean over targets of the shot-grouped k-fold R2."""
    fold = np.unique(groups, return_inverse=True)[1] % folds
    pred = np.full(Y.shape, np.nan)
    for f in range(folds):
        pred[fold == f] = ShapeMap.fit(X[fold != f], Y[fold != f], lam).predict(X[fold == f])
    return float(np.mean([r2(Y[:, j], pred[:, j]) for j in range(Y.shape[1])]))


def shape_fit(
    cache_root: Path = CACHE_ROOT, data_root: Path = DATA_ROOT, fit_shots: int = FIT_SHOTS, workers: int = 8
) -> None:
    """Fit the ShapeMap on `fit_shots` DIII-D train shots (scorer-derived targets) and report R2 on the 40 held-out
    scoring shots against the constant (train-mean) prediction. Saves shape_map.json in the cache root."""
    d = DiiidShape(cache_root, data_root, workers)
    train_rows, test_rows = d.targets["train_rows"], d.targets["test_rows"]
    sid = np.asarray(d.fr.shot_index)
    keep = np.isin(sid[train_rows], d.fit_shot_ids(fit_shots))
    if np.isin(sid[test_rows], sid[train_rows]).any():
        raise ValueError("held-out shots overlap the fit shots")
    rows, Y = train_rows[keep], target_matrix(d.targets["train"][keep])
    geom = d.geometry(rows)
    ok = np.isfinite(geom).all(1)
    X = d.features(rows[ok], geom[ok])
    Y, groups = Y[ok], sid[rows[ok]]
    cv = {lam: group_cv(X, Y, groups, lam) for lam in LAMS}
    lam = max(cv, key=cv.get)
    smap = ShapeMap.fit(X, Y, lam)
    smap.save(cache_root / MAP_FILE)
    logger.info(f"{len(X)} frames of {fit_shots} shots; grouped-CV mean R2 per lambda {cv} -> {lam}")
    test_geom = d.geometry(test_rows)
    test_geom = np.where(np.isfinite(test_geom), test_geom, np.nanmedian(test_geom, 0))
    pred, Yt = smap.predict(d.features(test_rows, test_geom)), target_matrix(d.targets["test"])
    for j, t in enumerate(TARGETS):
        const = r2(Yt[:, j], np.full(len(Yt), smap.intercept[j]))
        logger.info(f"held-out {t}: R2 {r2(Yt[:, j], pred[:, j]):+.3f} (constant train mean {const:+.3f})")


def plasma_fn(calib: Calibration, G: np.ndarray, fr: Frames, rows: np.ndarray, alpha: float) -> PlasmaFlux:
    ip = np.asarray(fr.ip[rows], np.float64)

    def flux(kappa: float, place: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
        sel = slice(None) if m is None else m
        shape = PlasmaShape(kappa, alpha)
        return calib.scale * calib.plasma_ratio * plasma_psi(G, fr.grid_R, fr.grid_Z, ip[sel], place, shape)

    return flux


def axis_drive(
    fr: Frames, rows: np.ndarray, shot_dir: Path, geom: np.ndarray, pred: np.ndarray, workers: int
) -> np.ndarray:
    """(N,) Shafranov drive eps (beta_p + li / 2) of frames `rows` (whole shots) from placement rows (R0, a), the
    ShapeMap's (Z_axis, kappa, ...) and the Thomson volume pressure."""
    kappa = pred[:, TARGETS.index("kappa")]
    pe = shot_pressure(fr, rows, shot_dir, workers, "volume", np.c_[geom[:, :2], kappa], pred[:, 0])
    return shafranov_drive(geom[:, 0], geom[:, 1], kappa, pe, np.asarray(fr.ip[rows], np.float64))


def cons_r2(y: np.ndarray, derived: np.ndarray) -> dict[str, float]:
    return {n: round(r2(y[:, j], derived[:, j]), 3) for j, n in enumerate(DERIVED[:-1])}


def shape_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    workers: int = 8,
    alpha: float = 2.0,
    shift_axis: bool = False,
    out: str = "shape_check.npz",
    psi_out: str = "",
) -> None:
    """Replay the MAST procedure on the 40 held-out DIII-D scoring shots (R0, a from labels in place of Thomson):
    fixed ellipse vs per-frame shape, each scored officially and per derived scalar against the labels. shift_axis
    also feeds the O-point back onto R0 + a * the ShiftModel's Delta / a (shift-fit). Arrays go to `out` in the cache
    root; psi_out also saves the shaped flux there (float16, as served) for li-check."""
    d = DiiidShape(cache_root, data_root, workers)
    fr, calib, rows = d.fr, d.calib, d.targets["test_rows"]
    smap = ShapeMap.load(cache_root / MAP_FILE)
    geom = d.geometry(rows)
    geom = np.where(np.isfinite(geom), geom, np.nanmedian(geom, 0))
    pred = smap.predict(d.features(rows, geom))
    r_target = None
    if shift_axis:
        x = axis_drive(fr, rows, d.shot_dir, geom, pred, workers)
        r_target = geom[:, 0] + geom[:, 1] * ShiftModel.load(cache_root / SHIFT_FILE).predict(x)
    scorer = Scorer("DIII-D")
    sign = AXIS_SIGN["DIII-D"]
    guard = coil_guard(fr.grid_R, fr.grid_Z, coils_of(d.shot_dir / fr.shots[0]["file"]))
    ladder = ShapeLadder(Deriver(fr.grid_R, fr.grid_Z, "DIII-D", sign), guard, workers)
    G = cached_grid_greens(cache_root / "diii_d_train", fr.grid_R, fr.grid_Z)
    psi, recs = np.empty((len(rows), len(fr.grid_Z), len(fr.grid_R)), np.float32), []
    fixed = np.empty_like(psi)
    for i, r in blocks(rows):
        base = calib.apply(d.vacuum(r), 0.0)
        flux = plasma_fn(calib, G, fr, r, alpha)
        r_axis = None if r_target is None else r_target[i]
        psi[i], rec = ladder.run(base, flux, geom[i], pred[i, 0], pred[i, 1], r_axis)
        fixed[i] = ladder.candidate(base, flux, np.c_[geom[i, :2], np.zeros(len(r))], 1.6)
        recs.append(rec)
    if psi_out:
        np.save(cache_root / psi_out, psi.astype(SERVED))
    picks = np.concatenate([r["pick"] for r in recs])
    logger.info(f"pick shares {np.bincount(picks, minlength=1 + len(KAPPA_LADDER)) / len(picks)}")
    masked = Deriver(fr.grid_R, fr.grid_Z, "DIII-D", sign, (scorer.mask_coarse, scorer.mask_f), with_li=True)
    y = d.targets["test"]
    mask = np.isin(np.arange(len(fr.shot_index)), rows)
    saved = {"target": y, "pred": pred, "pick": picks, "place": np.concatenate([r["place"] for r in recs])}
    for name, p in (("fixed", fixed), ("shaped", psi)):
        saved[name] = masked.many(p, workers)
        logger.info(f"{name}: cons R2 {cons_r2(y, saved[name])}")
        logger.info(f"{name}: {summary_line(score_rows(fr, p, mask, cache_root))}")
    np.savez(cache_root / out, **saved)


def score_rows(fr: Frames, psi: np.ndarray, mask: np.ndarray, cache_root: Path) -> dict[str, Any]:
    return score_frames(fr, psi, mask=mask, cache_dir=cache_root / "refs")


@dataclass
class MastShapeBuild:
    """The MAST candidate: the current entry's flux (fixed ellipse, solenoid-shape correction) with the plasma's
    elongation and centre height chosen per frame. Reads MAST inputs and geometry only."""

    cache: MastCache
    smap: ShapeMap | None = None
    alpha: float = 2.0
    workers: int = 8
    shift: ShiftModel | None = None

    @cached_property
    def coils(self) -> dict[str, np.ndarray]:
        return coils_of(list_shots(self.cache.data_root, self.cache.config)[0])

    @cached_property
    def sol(self) -> np.ndarray:
        fr = self.cache.fr
        return coil_basis(fr.grid_R, fr.grid_Z, self.coils, list(self.coils["columns"]))[SOLENOID_COLUMN]

    @cached_property
    def geom(self) -> np.ndarray:
        """(N, 3) R0, a, Z0 = 0 from the Thomson density extent (mast-physics --frac 0.4)."""
        return np.load(self.cache.dir / "plasma_geom.npy").astype(np.float64)

    def vacuum(self, rows: np.ndarray) -> np.ndarray:
        """Raw vacuum flux with the solenoid at half its shipped current (the shape the current entry uses)."""
        i_sol = np.asarray(self.cache.fr.currents[rows, SOLENOID_COLUMN], np.float32)
        return np.asarray(self.cache.fr.psi_vac[rows], np.float32) - 0.5 * i_sol[:, None, None] * self.sol

    def solenoid_shape(self, rows: np.ndarray) -> np.ndarray:
        """The current entry's correction: half the solenoid's calibrated flux less its grid mean (level kept)."""
        fr, calib = self.cache.fr, self.cache.calib
        corr = calib.scale * 0.5 * np.asarray(fr.currents[rows, SOLENOID_COLUMN], np.float32)[:, None, None] * self.sol
        return corr - corr.mean((1, 2), keepdims=True)

    def base(self, rows: np.ndarray) -> np.ndarray:
        """The current entry's flux without its plasma term."""
        return self.cache.calib.apply(np.asarray(self.cache.fr.psi_vac[rows], np.float32), 0.0) - self.solenoid_shape(
            rows
        )

    def targets(self) -> np.ndarray:
        fr, rows = self.cache.fr, np.arange(len(self.geom))
        g = self.geom
        X = np.concatenate(
            [features(self.vacuum(r), fr.grid_R, fr.grid_Z, g[i, 0], g[i, 1], fr.ip[r]) for i, r in blocks(rows)]
        )
        if self.smap is None:
            raise ValueError("no ShapeMap: run shape-fit first")
        clipped = ((X < self.smap.x_lo) | (X > self.smap.x_hi)).mean(0)
        logger.info(
            f"MAST features outside the DIII-D range (clipped): {dict(zip(FEATURES, clipped.round(3), strict=True))}"
        )
        return self.smap.predict(X)

    def r_target(self, pred: np.ndarray) -> np.ndarray | None:
        """(N,) the Shafranov-shifted axis R0 + a * Delta / a from the Thomson placement, or None with no ShiftModel."""
        if self.shift is None:
            return None
        mast = self.cache.data_root / "data" / self.cache.config
        x = axis_drive(self.cache.fr, np.arange(len(pred)), mast, self.geom, pred, self.workers)
        return self.geom[:, 0] + self.geom[:, 1] * self.shift.predict(x)

    def build(self, out: str) -> dict[str, np.ndarray]:
        fr = self.cache.fr
        G = cached_grid_greens(self.cache.dir, fr.grid_R, fr.grid_Z)
        guard = coil_guard(fr.grid_R, fr.grid_Z, self.coils)
        ladder = ShapeLadder(Deriver(fr.grid_R, fr.grid_Z, "MAST", MAST_AXIS_SIGN), guard, self.workers)
        pred = self.targets()
        r_target = self.r_target(pred)
        shape = (len(pred), len(fr.grid_Z), len(fr.grid_R))
        psi = np.lib.format.open_memmap(self.cache.dir / out, "w+", np.float16, shape)
        recs = []
        for i, r in blocks(np.arange(len(pred))):
            flux = plasma_fn(self.cache.calib, G, fr, r, self.alpha)
            r_axis = None if r_target is None else r_target[i]
            p, rec = ladder.run(self.base(r), flux, self.geom[i], pred[i, 0], pred[i, 1], r_axis)
            psi[i] = p.astype(np.float16)
            recs.append(rec)
            logger.info(f"{out}: {i[-1] + 1}/{len(pred)} frames")
        psi.flush()
        record = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]} | {"target": pred}
        np.savez(self.cache.dir / out.replace(".npy", "_record.npz"), **record)
        return record


def mast_shape(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    out: str = "psi_shape_dmap.npy",
    config: str = "mast_public_test",
    alpha: float = 2.0,
    workers: int = 8,
    shift_axis: bool = False,
) -> None:
    """Build `out` in the MAST cache: per-frame plasma elongation and centre height from the DIII-D ShapeMap, gated
    label-free; the record (picks, targets, derived scalars before / after) goes beside it. shift_axis also feeds the
    O-point back onto the DIII-D ShiftModel's Shafranov-shifted axis from the Thomson placement."""
    shift = ShiftModel.load(cache_root / SHIFT_FILE) if shift_axis else None
    build = MastShapeBuild(
        MastCache(cache_root, data_root, config), ShapeMap.load(cache_root / MAP_FILE), alpha, workers, shift
    )
    rec = build.build(out)
    shot = np.asarray(build.cache.fr.shot_index)
    picks = np.bincount(rec["pick"], minlength=1 + len(KAPPA_LADDER)) / len(rec["pick"])
    logger.info(
        f"{out}: pick shares (fixed, ladder...) {picks.round(3)}; no candidate passed {1 - rec['any_ok'].mean():.3f}"
    )
    for name, d, g in (("fixed", rec["derived_fixed"], build.geom), ("shaped", rec["derived"], rec["place"])):
        logger.info(f"{name}: {json.dumps(summary(d, shot, g))}")


def derive_check(
    psi_files: str = "psi_k1.6_a2.0_f0.4.npy",
    config: str = "mast_public_test",
    every: int = 10,
    workers: int = 8,
    cache_root: Path = CACHE_ROOT,
    out: str = "",
) -> None:
    """Label-free table for each comma-separated flux file in the MAST cache: the scorer's derived scalars on every
    `every`-th frame (the same frames for each file), MAST orientation, no envelope mask. A file named '<npy>-solshape'
    applies the current entry's solenoid-shape correction on the fly."""
    cache = MastCache(cache_root, config=config)
    rows = np.arange(0, len(cache.fr.shot_index), every)
    deriver = Deriver(cache.fr.grid_R, cache.fr.grid_Z, "MAST", MAST_AXIS_SIGN, with_li=True)
    geom = np.load(cache.dir / "plasma_geom.npy")[rows]
    table = {}
    for name in psi_files.split(","):
        psi = load_flux(cache, name, rows)
        table[name] = summary(deriver.many(psi, workers), np.asarray(cache.fr.shot_index[rows]), geom)
        logger.info(f"{name} ({len(rows)} frames): {json.dumps(table[name])}")
    if out:
        Path(out).write_text(json.dumps(table, indent=1))


def load_flux(cache: MastCache, name: str, rows: np.ndarray) -> np.ndarray:
    file, _, tag = name.partition("-")
    psi = np.asarray(np.load(cache.dir / file, mmap_mode="r")[rows], np.float32)
    return psi - MastShapeBuild(cache).solenoid_shape(rows) if tag == "solshape" else psi


def shift_inputs(d: DiiidShape, smap: ShapeMap, split: str, keep: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Drive, Delta / a and the label geometry of DIII-D cons_targets `split` frames (optionally only `keep`): Delta is
    the scorer-derived R_axis of EFIT minus the labelled boundary centre, NaN where the label boundary is missing."""
    rows, y = d.targets[f"{split}_rows"], d.targets[split][:, CONS_INDEX["R_axis"]]
    if keep is not None:
        rows, y = rows[keep], y[keep]
    sid = np.asarray(d.fr.shot_index)
    whole = np.flatnonzero(np.isin(sid, sid[rows]))  # the pressure reads whole shots; targets skip some frames
    at = np.searchsorted(whole, rows)
    geom = d.geometry(whole)
    bad = ~np.isfinite(geom).all(1)
    geom = np.where(bad[:, None], np.nanmedian(geom, 0), geom)
    x = axis_drive(d.fr, whole, d.shot_dir, geom, smap.predict(d.features(whole, geom)), d.workers)[at]
    geom, bad = geom[at], bad[at]
    return {"x": x, "delta": np.where(bad, np.nan, (y - geom[:, 0]) / geom[:, 1]), "geom": geom, "r_axis": y}


def shift_fit(
    cache_root: Path = CACHE_ROOT, data_root: Path = DATA_ROOT, fit_shots: int = FIT_SHOTS, workers: int = 8
) -> None:
    """Fit the ShiftModel on the ShapeMap's DIII-D fit shots (the drive from the label (R0, a), the map's kappa and
    Thomson pressure) and report R_axis = R0 + a * Delta / a on the 40 held-out scoring shots against the constant
    mean Delta / a. Saves shift_model.json in the cache root. DIII-D labels only."""
    d = DiiidShape(cache_root, data_root, workers)
    smap = ShapeMap.load(cache_root / MAP_FILE)
    sid = np.asarray(d.fr.shot_index)
    train = shift_inputs(d, smap, "train", np.isin(sid[d.targets["train_rows"]], d.fit_shot_ids(fit_shots)))
    model = ShiftModel.fit(train["x"], train["delta"])
    model.save(cache_root / SHIFT_FILE)
    test = shift_inputs(d, smap, "test")
    r0, a = test["geom"][:, 0], test["geom"][:, 1]
    const, fit = float(np.nanmean(train["delta"])), model.predict(test["x"])
    logger.info(
        f"{model} on {np.isfinite(train['delta']).sum()} frames; held-out Delta / a R2 {r2(test['delta'], fit):+.3f}, "
        f"R_axis R2 {r2(test['r_axis'], r0 + a * fit):+.3f} vs constant Delta / a {const:.3f}: "
        f"{r2(test['r_axis'], r0 + a * const):+.3f}"
    )


def replay_rows(ev: PhysicsEval, shots: ReplayShots, exclude: np.ndarray) -> np.ndarray:
    """Frames of the scored held-out shots, or of the other held-out shots less those in `exclude` (the ShapeMap's
    fit shots, so the map is scored out of sample)."""
    if shots == "scoring":
        return ev.rows
    return np.nonzero(ev.held & ~ev.subset & ~np.isin(ev.fr.shot_index, exclude))[0]


def geom_scal_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    shots: str = "scoring",
    perturb_seed: int = -1,
    frame_sd: float = 0.04,
    a_scale: float = 1.0,
    kappa: float = 1.6,
    alpha: float = 2.0,
    eps_max: float = 0.65,
    workers: int = 8,
) -> None:
    """MAST q95 / betaN procedures replayed on held-out DIII-D shots and scored against their labels: the placement
    (R0, a) with the ShapeMap's kappa and triangularity (no boundary extraction), against the extraction variants of
    region-check on the physics flux placed the same way. The placement is the label (R0, a) standing in for MAST's
    Thomson extent, or with perturb_seed >= 0 that placement with a Thomson-like error (perturb_placement, per-frame
    a jitter frame_sd: MAST's Thomson a jumps by ~0.12 rms between frames, label-free); a_scale
    multiplies the placed a for a systematic Thomson-against-LCFS bias."""
    d = DiiidShape(cache_root, data_root, workers)
    ev = PhysicsEval.load(cache_root, data_root=data_root)
    replay = ScalarReplay(ev, replay_rows(ev, shots, d.fit_shot_ids()), eps_max, workers)
    geom = diii_d_geometry(ev.fr, replay.rows, ev.shot_dir)
    place = geom if perturb_seed < 0 else perturb_placement(geom, replay.shot_index, perturb_seed, frame_sd=frame_sd)
    place = np.c_[place[:, 0], place[:, 1] * a_scale, np.zeros(len(place))]
    smap = ShapeMap.load(cache_root / MAP_FILE)
    pred = smap.predict(d.features(replay.rows, place))
    const = np.broadcast_to(np.asarray(smap.intercept), pred.shape)
    variants = {
        "label boundary (upper bound)": replay.predict(geom[:, [0, 1, 3, 4]]),
        "A: placement + map kappa, tri": replay.predict(placement_shape(place, pred), pred[:, 0]),
        "A, tri 0": replay.predict(placement_shape(place, pred, use_tri=False), pred[:, 0]),
        "placement + constant map shape": replay.predict(placement_shape(place, const), const[:, 0]),
    }
    psi = replay.physics_flux(place, PlasmaShape(kappa, alpha))
    variants |= {name: p for name, (p, _) in replay.extraction(psi, place).items()}
    n_shots = len(np.unique(replay.shot_index))
    tag = (
        f"{shots} ({n_shots} shots, {len(replay.rows)} frames), seed {perturb_seed}, frame sd {frame_sd}, a x {a_scale}"
    )
    for name, p in variants.items():
        q, b = replay.r2(p)
        logger.info(f"[{tag}] {name}: R2 q95 {q:+.3f} betaN {b:+.3f} mean {(q + b) / 2:+.3f}")
    keys = {f"v{i}": p for i, p in enumerate(variants.values())}
    np.savez(
        cache_root / f"geom_scal_check_{shots}_{perturb_seed}.npz", truth=replay.truth, names=list(variants), **keys
    )


def mast_geom_scalars(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    out: str = "scal_vol_eps065_geom.npy",
    config: str = "mast_public_test",
    eps_max: float = 0.65,
    use_tri: bool = True,
    workers: int = 8,
) -> None:
    """MAST q95 / betaN with no boundary extraction: the Thomson (R0, a) of plasma_geom.npy and the DIII-D ShapeMap's
    kappa and triangularity from the vacuum field (mast-shape's features), volume pressure on ellipses at the map's
    Z_axis. No MAST labels, no envelope."""
    smap = ShapeMap.load(cache_root / MAP_FILE)
    build = MastShapeBuild(MastCache(cache_root, data_root, config), smap, workers=workers)
    pred = build.targets()
    shape = placement_shape(build.geom, pred, use_tri)
    build.cache.shape_scalars(shape, eps_max, out, workers, "volume", pred[:, 0])


COMMANDS = [shape_fit, shape_check, mast_shape, derive_check, geom_scal_check, mast_geom_scalars, shift_fit]
