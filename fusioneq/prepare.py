"""Turn a config's parquet shots into flat per-frame arrays on disk (one pass, parallel)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from fusioneq.data import HR_STATIONS, LABEL_COLUMNS, list_shots, load_shot, read_row, thomson_features
from fusioneq.greens import coil_basis, vacuum_flux


def process_one(path: Path) -> dict[str, np.ndarray | str | int]:
    shot = load_shot(path)
    basis = coil_basis(shot.grid_R, shot.grid_Z, shot.coils, shot.coil_names)
    covered = np.asarray([c in set(np.asarray(shot.coils["coil_input_column"]).astype(str)) for c in shot.coil_names])
    out = {
        "file": path.name,
        "T": len(shot.efit_times),
        "efit_times": shot.efit_times.astype(np.float32),
        "currents": shot.currents,
        "ip": shot.ip,
        "thomson": shot.thomson,
        "psi_vac": vacuum_flux(basis[covered], shot.currents[:, covered]).astype(np.float16),
        "grid_R": shot.grid_R.astype(np.float32),
        "grid_Z": shot.grid_Z.astype(np.float32),
    }
    if shot.psi is not None:
        out["psi"] = shot.psi.astype(np.float16)
        out["scalars"] = shot.scalars
    return out


def prepare(data_root: Path, config: str, out_dir: Path, limit: int | None = None, workers: int = 12) -> Path:
    paths = list_shots(data_root, config)[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{config}: {len(paths)} shots -> {out_dir}")
    parts: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for i, part in enumerate(ex.map(process_one, paths, chunksize=4)):
            parts.append(part)
            if (i + 1) % 250 == 0:
                logger.info(f"  {i + 1}/{len(paths)}")
    keys = ["efit_times", "currents", "ip", "thomson", "psi_vac"] + (["psi", "scalars"] if "psi" in parts[0] else [])
    for k in keys:
        np.save(out_dir / f"{k}.npy", np.concatenate([p[k] for p in parts], axis=0))
    shot_index = np.concatenate([np.full(p["T"], i, np.int32) for i, p in enumerate(parts)])
    np.save(out_dir / "shot_index.npy", shot_index)
    np.save(out_dir / "grid_R.npy", parts[0]["grid_R"])
    np.save(out_dir / "grid_Z.npy", parts[0]["grid_Z"])
    (out_dir / "shots.json").write_text(json.dumps([{"file": p["file"], "T": p["T"]} for p in parts]))
    logger.info(f"{config}: {len(shot_index)} frames written")
    return out_dir


def thomson_hr_one(path: Path, window: float) -> np.ndarray:
    row = read_row(path, exclude=LABEL_COLUMNS)
    return thomson_features(row, np.asarray(row["efit_times"], float), HR_STATIONS[row["source"]], window)


def prepare_thomson_hr(data_root: Path, config: str, cache_dir: Path, window: float, workers: int = 12) -> Path:
    """Write cache_dir/thomson_hr.npy in the cache's own shot order; the existing cache files are left untouched."""
    shots = json.loads((cache_dir / "shots.json").read_text())
    by_name = {p.name: p for p in list_shots(data_root, config)}
    paths = [by_name[s["file"]] for s in shots]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(thomson_hr_one, paths, [window] * len(paths), chunksize=4))
    bad = [s["file"] for s, part in zip(shots, parts, strict=True) if len(part) != s["T"]]
    if bad:
        raise ValueError(f"{len(bad)} shots disagree with the cache's frame counts, e.g. {bad[:3]}")
    out = cache_dir / "thomson_hr.npy"
    np.save(out, np.concatenate(parts))  # thomson_features already returns float32
    logger.info(f"{config}: thomson_hr {sum(len(p) for p in parts)} frames x {parts[0].shape[1]} features -> {out}")
    return out
