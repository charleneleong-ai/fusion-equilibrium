"""The li lever: reshape the core flux toward a target internal inductance, LCFS kept (fusioneq/li_shape.py).

li-check  the shifted MAST procedure's DIII-D replay flux (shape-check --psi-out) remapped per setting and scored
          officially against the DIII-D labels
mast-li   the MAST candidate flux: the served flux remapped to level + k (li - median li), no MAST labels, no envelope
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from fusioneq.commands.shape import BLOCK, DiiidShape, score_rows
from fusioneq.li_shape import C_GRID, CoreShaper, shot_smooth, shrink_targets
from fusioneq.mast_shape import SERVED
from fusioneq.physics import MAST_AXIS_SIGN, MastCache
from fusioneq.score import AXIS_SIGN, summary_line
from fusioneq.settings import CACHE_ROOT, DATA_ROOT
from fusioneq.shape_map import r2

LI_COL = 6  # li in cons_targets / DERIVED
C_ZERO = int(np.argmin(np.abs(np.asarray(C_GRID))))


def remapped(psi: np.ndarray, bumps: np.ndarray, c: np.ndarray) -> np.ndarray:
    """psi + c * bump per frame, rounded to the served dtype."""
    return (np.asarray(psi, np.float32) + c[:, None, None] * np.asarray(bumps, np.float32)).astype(SERVED)


def parse_settings(settings: str) -> list[tuple[float | None, float]]:
    """'level:k,...' -> [(level, k)]; level 'own' keeps our own median li (shrink only)."""
    out = []
    for item in settings.split(","):
        level, k = item.split(":")
        out.append((None if level == "own" else float(level), float(k)))
    return out


def li_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    psi_file: str = "shape_check_shift_psi.npy",
    settings: str = "own:1,own:0.5,own:0.25,1.05:0.5,1.05:0.25,1.05:0,0.95:0.25,1.15:0.25",
    smooth: int = 0,
    workers: int = 8,
    out: str = "li_check.npz",
) -> None:
    """Score the replay flux remapped to target li = level + k (li - median li) for each 'level:k' setting on the 40
    held-out DIII-D scoring shots ('own' level: our own median); smooth > 1 takes li's within-shot rolling median over
    that many frames first. DIII-D labels only."""
    d = DiiidShape(cache_root, data_root, workers)
    fr, rows = d.fr, d.targets["test_rows"]
    psi = np.load(cache_root / psi_file).astype(np.float32)
    shaper = CoreShaper(fr.grid_R, fr.grid_Z, "DIII-D", AXIS_SIGN["DIII-D"])
    bumps, curves = shaper.many(psi, workers)
    li0, truth = curves[:, C_ZERO], d.targets["test"][:, LI_COL]
    ok = np.isfinite(li0) & np.isfinite(truth)
    logger.info(
        f"replay li (unmasked) p5/50/95 {np.percentile(li0[ok], [5, 50, 95]).round(3)} vs truth "
        f"{np.percentile(truth[ok], [5, 50, 95]).round(3)}; corr {np.corrcoef(li0[ok], truth[ok])[0, 1]:.3f}; "
        f"no curve on {np.mean(~np.isfinite(li0)):.3f}"
    )
    mask = np.isin(np.arange(len(fr.shot_index)), rows)
    logger.info(f"as built: {summary_line(score_rows(fr, psi, mask, cache_root))}")
    saved = {"li0": li0, "truth": truth, "curves": curves}
    li_s = shot_smooth(li0, np.asarray(fr.shot_index[rows]), smooth)
    for level, k in parse_settings(settings):
        target = shrink_targets(li_s, float(np.nanmedian(li0)) if level is None else level, k)
        c = shaper.coefficient(curves, target)
        hit = np.array([np.interp(ci, C_GRID, cu) for ci, cu in zip(c, curves, strict=True)])
        res = score_rows(fr, remapped(psi, bumps, c).astype(np.float32), mask, cache_root)
        logger.info(
            f"level {level} k {k} smooth {smooth}: target R2 (unmasked) {r2(truth[ok], hit[ok]):+.3f}, c p5/50/95 "
            f"{np.percentile(c, [5, 50, 95]).round(3)}; {summary_line(res)}"
        )
        saved[f"c_{level}_{k}"] = c
    np.savez(cache_root / out, **saved)


def mast_curves(cache: MastCache, psi_file: str, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """(bumps, curves) of a MAST flux file, computed once and kept beside it (<stem>_li_bumps / _li_curves.npy)."""
    stem = cache.dir / Path(psi_file).stem
    b_path, c_path = Path(f"{stem}_li_bumps.npy"), Path(f"{stem}_li_curves.npy")
    if b_path.exists() and c_path.exists():
        return np.load(b_path, mmap_mode="r"), np.load(c_path)
    src = np.load(cache.dir / psi_file, mmap_mode="r")
    shaper = CoreShaper(cache.fr.grid_R, cache.fr.grid_Z, "MAST", MAST_AXIS_SIGN)
    bumps = np.lib.format.open_memmap(b_path.with_suffix(".tmp.npy"), "w+", SERVED, src.shape)
    curves = np.full((len(src), len(C_GRID)), np.nan)
    for s in range(0, len(src), BLOCK):
        sl = slice(s, min(s + BLOCK, len(src)))
        b, curves[sl] = shaper.many(np.asarray(src[sl], np.float32), workers)
        bumps[sl] = b.astype(SERVED)
        logger.info(f"{psi_file}: li curves {sl.stop}/{len(src)}")
    bumps.flush()
    np.save(c_path, curves)
    b_path.with_suffix(".tmp.npy").rename(b_path)
    return np.load(b_path, mmap_mode="r"), curves


def mast_li(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    psi_file: str = "psi_shape_dmap_shift.npy",
    out: str = "psi_shape_dmap_shift_li.npy",
    level: float = 0.85,
    k: float = 0.25,
    smooth: int = 0,
    config: str = "mast_public_test",
    workers: int = 8,
) -> None:
    """Build `out` in the MAST cache: `psi_file` with its core remapped to target li = level + k (li - median li). The
    level is a published MAST value, k is set on the DIII-D replay; the median is our own flux's (no labels). The
    record (li before / after, c, curves) goes beside it. smooth > 1: li's within-shot rolling median first."""
    cache = MastCache(cache_root, data_root, config)
    fr = cache.fr
    src = np.load(cache.dir / psi_file, mmap_mode="r")
    bumps, curves = mast_curves(cache, psi_file, workers)
    li0 = curves[:, C_ZERO]
    target = shrink_targets(shot_smooth(li0, np.asarray(fr.shot_index), smooth), level, k)
    c = CoreShaper(fr.grid_R, fr.grid_Z, "MAST", MAST_AXIS_SIGN).coefficient(curves, target)
    psi = np.lib.format.open_memmap(cache.dir / out, "w+", SERVED, src.shape)
    for s in range(0, len(src), BLOCK):
        sl = slice(s, min(s + BLOCK, len(src)))
        psi[sl] = remapped(src[sl], bumps[sl], c[sl])
    psi.flush()
    hit = np.array([np.interp(ci, C_GRID, cu) for ci, cu in zip(c, curves, strict=True)])
    np.savez(cache.dir / out.replace(".npy", "_record.npz"), li0=li0, target=target, c=c, li=hit, curves=curves)
    f, edge = np.isfinite(li0), np.mean((c <= C_GRID[0]) | (c >= C_GRID[-1]))
    logger.info(
        f"{out}: level {level} k {k} smooth {smooth}, median li {np.nanmedian(li0):.3f}; li p5/50/95 "
        f"{np.percentile(li0[f], [5, 50, 95]).round(3)} -> {np.percentile(hit[f], [5, 50, 95]).round(3)}; "
        f"c p5/50/95 {np.percentile(c, [5, 50, 95]).round(3)}; at the grid edge {edge:.3f}; "
        f"no curve {np.mean(~f):.4f}"
    )


COMMANDS = [li_check, mast_li]
