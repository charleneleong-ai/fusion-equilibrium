"""One shot in, aligned per-frame arrays out: inputs resampled onto efit_times, targets untouched."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from vendor.starter.data_fixes import align_mast_thomson_core, fix_d3d_ip_times

D3D_COILS = ["ECOILA", "bcoil"] + [f"F{i}{s}" for i in range(1, 10) for s in "AB"]
MAST_COILS = ["sol_current", "tf_current", "efps_current"] + [f"p{i}{s}_current" for i in range(2, 7) for s in "lu"]
SCALARS = ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis")
# Fixed spatial stations per machine, measured from the shipped chord geometry (see README "Thomson").
STATIONS = {
    "DIII-D": {
        "core": np.linspace(0.05, 0.80, 12),
        "edge": np.linspace(1.70, 2.25, 8),
    },  # core: Z on a vertical chord; edge: R
    "MAST": {"core": np.linspace(0.30, 1.45, 12), "edge": np.linspace(1.33, 1.49, 8)},  # both midplane, coordinate is R
}
# One station per DIII-D chord (44 core, 10 edge): the profile shape that sets li survives the resampling.
HR_STATIONS = {"DIII-D": {"core": np.linspace(0.05, 0.80, 44), "edge": np.linspace(1.70, 2.25, 10)}}


@dataclass
class Shot:
    source: str
    efit_times: np.ndarray  # (T,)
    grid_R: np.ndarray  # (65,)
    grid_Z: np.ndarray  # (65,)
    coil_names: list[str]  # magnetics column per current feature
    currents: np.ndarray  # (T, C) kA or kA-turn, on efit_times
    ip: np.ndarray  # (T,) kA
    thomson: np.ndarray  # (T, 4 * n_stations) Te/ne core + edge on fixed stations, log-scaled
    coils: dict[str, np.ndarray] = field(default_factory=dict)  # geometry columns (C,)
    psi: np.ndarray | None = None  # (T, 65, 65)
    scalars: np.ndarray | None = None  # (T, 5) q95, betaN, li, r_axis, z_axis


# Equilibrium labels: by far the largest columns, and unused by anything that only needs the inputs.
LABEL_COLUMNS = ("efit_psirz", "efit_lcfs_r", "efit_lcfs_z")


def read_row(path: Path, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    columns = [c for c in pq.read_schema(path).names if c not in exclude] if exclude else None
    return pq.read_table(path, columns=columns).slice(0, 1).to_pylist()[0]


def interp_series(t_out: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 2:
        return np.zeros_like(t_out)
    return np.interp(t_out, t[ok], v[ok])


def coil_columns(row: dict) -> list[str]:
    return [f"magnetics_{c}" for c in (D3D_COILS if row["source"] == "DIII-D" else MAST_COILS)]


def currents_on_efit(row: dict, efit_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag_t = np.asarray(row["magnetics_time"], float)
    cols = coil_columns(row)
    cur = np.column_stack([interp_series(efit_t, mag_t, row[c]) for c in cols]).astype(np.float32)
    ip_t = fix_d3d_ip_times(row) if row["source"] == "DIII-D" else mag_t
    ip = interp_series(efit_t, ip_t, row["magnetics_plasma_current"]).astype(np.float32)
    return cur, ip


def profile_on_efit(times: np.ndarray, prof: np.ndarray, efit_t: np.ndarray, window: float = 0.0) -> np.ndarray:
    """Thomson profile per EFIT frame, prof: (Tt, C), NaN channels left as NaN. window = 0 takes the nearest sample;
    window > 0 averages each channel's finite samples within +-window/2, falling back to the nearest where none."""
    idx = np.clip(np.searchsorted(times, efit_t), 0, len(times) - 1)
    left = np.clip(idx - 1, 0, len(times) - 1)
    pick = np.where(np.abs(times[left] - efit_t) < np.abs(times[idx] - efit_t), left, idx)
    nearest = prof[pick]
    if window <= 0:
        return nearest
    lo = np.searchsorted(times, efit_t - window / 2)
    hi = np.searchsorted(times, efit_t + window / 2, side="right")
    ok = np.isfinite(prof)
    zero = np.zeros((1, prof.shape[1]))
    total = np.vstack([zero, np.cumsum(np.where(ok, prof, 0.0), axis=0)])
    count = np.vstack([zero, np.cumsum(ok, axis=0)])
    n = count[hi] - count[lo]
    return np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), nearest)


def resample_profile(coord: np.ndarray, prof: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """Per frame, interpolate a spatial profile onto fixed stations; NaN-safe, zero outside coverage."""
    order = np.argsort(coord)
    out = np.zeros((prof.shape[0], len(stations)), np.float32)
    for k in range(prof.shape[0]):
        v = prof[k][order]
        ok = np.isfinite(v)
        if ok.sum() >= 2:
            out[k] = np.interp(stations, coord[order][ok], v[ok], left=0.0, right=0.0)
    return out


def thomson_geometry(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(core_pos, core_Te, core_ne, edge_pos): the informative coordinate of each system plus aligned profiles."""
    core_Te = np.stack([np.asarray(p, float) for p in row["thomson_core_Te"]])
    core_ne = np.stack([np.asarray(p, float) for p in row["thomson_core_ne"]])
    if row["source"] == "DIII-D":
        names = np.asarray(row["thomson_chord_name"]).astype(str)
        is_core = np.array([n.startswith("TS_core") for n in names])
        is_tang = np.array([n.startswith("TS_tangential") for n in names])
        core_pos = np.asarray(row["thomson_chord_Z"], float)[is_core]
        edge_pos = np.asarray(row["thomson_chord_R"], float)[is_tang]
        if len(core_pos) != core_Te.shape[1]:
            core_pos = np.linspace(0.05, 0.80, core_Te.shape[1])
    else:
        core_pos, core_Te, core_ne = align_mast_thomson_core(row)
        edge_pos = np.asarray(row["thomson_edge_spatial"], float)
    if len(edge_pos) != len(row["thomson_edge_Te"][0]):
        edge_pos = np.asarray(row["thomson_edge_spatial"], float)
    return core_pos, core_Te, core_ne, edge_pos


DIII_D_CORE_R = 1.94  # the DIII-D core Thomson chord's major radius, where the shot file lacks it


def core_chord_rz(row: dict[str, Any], pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, Z) of each core Thomson point (thomson_geometry's `pos`): a vertical chord at fixed R on DIII-D, the
    midplane on MAST."""
    if row["source"] != "DIII-D":
        return pos, np.zeros_like(pos)
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    R = np.asarray(row["thomson_chord_R"], float)[np.char.startswith(names, "TS_core")]
    return (R if len(R) == len(pos) else np.full_like(pos, DIII_D_CORE_R)), pos


def thomson_features(
    row: dict[str, Any], efit_t: np.ndarray, stations: dict[str, np.ndarray] | None = None, window: float = 0.0
) -> np.ndarray:
    """log1p(Te [eV]) and log1p(ne [1e19 m^-3]) on fixed stations, core then edge, Te then ne -> (T, 2 * n_stations).
    Defaults: STATIONS (12 core, 8 edge -> 40 features) and the nearest Thomson sample per frame."""
    core_pos, core_Te, core_ne, edge_pos = thomson_geometry(row)
    st = stations or STATIONS[row["source"]]
    edge_Te = np.stack([np.asarray(p, float) for p in row["thomson_edge_Te"]])
    edge_ne = np.stack([np.asarray(p, float) for p in row["thomson_edge_ne"]])
    systems = (
        (np.asarray(row["thomson_core_times"], float), core_pos, core_Te, core_ne, st["core"]),
        (np.asarray(row["thomson_edge_times"], float), edge_pos, edge_Te, edge_ne, st["edge"]),
    )
    feats = []
    for times, pos, Te, ne, stations in systems:
        Te_f = profile_on_efit(times, Te, efit_t, window)
        ne_f = profile_on_efit(times, ne, efit_t, window)
        feats.append(np.log1p(np.clip(resample_profile(pos, Te_f, stations), 0, None)))
        feats.append(np.log1p(np.clip(resample_profile(pos, ne_f, stations) / 1e19, 0, None)))
    return np.concatenate(feats, axis=1).astype(np.float32)


def load_shot(path: Path, with_targets: bool = True) -> Shot:
    row = read_row(path)
    efit_t = np.asarray(row["efit_times"], float)
    cur, ip = currents_on_efit(row, efit_t)
    shot = Shot(
        source=row["source"],
        efit_times=efit_t,
        grid_R=np.asarray(row["efit_grid_R"], float),
        grid_Z=np.asarray(row["efit_grid_Z"], float),
        coil_names=coil_columns(row),
        currents=cur,
        ip=ip,
        thomson=thomson_features(row, efit_t),
        coils={k: np.asarray(row[k]) for k in row if k.startswith("coil_")},
    )
    if with_targets and "efit_psirz" in row:
        shot.psi = np.stack([np.asarray(f, np.float32) for f in row["efit_psirz"]])
        shot.scalars = np.column_stack([np.asarray(row[s], np.float32) for s in SCALARS])
    return shot


def list_shots(data_root: Path, config: str) -> list[Path]:
    """HF streams a config's parquet files in sorted name order; submissions must follow it."""
    return sorted((data_root / "data" / config).glob("*.parquet"))
