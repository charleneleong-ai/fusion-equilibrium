"""Machine-agnostic q95 and betaN from the plasma current, the TF field, the plasma boundary and Thomson pressure.

q95   = c_q * 5 a^2 B0 / (R0 Ip) * (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)) / 2 * (1.17 - 0.65 eps) / (1 - eps^2)^2
betaN = c_b * beta_t[%] * a B0 / Ip,  beta_t = 2 mu0 p_e,peak / B0^2      (Ip in MA)
B0    = mu0 N_tf |I_tf| / (2 pi R0): the TF column carries per-turn kA and N_tf is each machine's published turn count.

c_q and c_b are the only fitted numbers (log-mean ratio to EFIT on DIII-D train shots, where labels are allowed); c_b
absorbs ion pressure and the peak-to-volume-average profile factor. q95 ~ 1/Ip diverges on ramp frames, so it is capped
at the DIII-D 99th percentile. On MAST the boundary comes from our predicted flux.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from fusioneq.data import LABEL_COLUMNS, core_chord_rz, profile_on_efit, read_row, thomson_geometry
from fusioneq.greens import MU0
from fusioneq.plasma import ellipse_rho2

EV = 1.602176634e-19
TF_TURNS = {"DIII-D": 144, "MAST": 24}
TF_COLUMN = 1  # data.D3D_COILS[1] = bcoil, data.MAST_COILS[1] = tf_current
MIN_IP_MA = 0.05
RHO = np.linspace(0.0, 1.0, 41)  # normalised minor radius grid for the volume average
PressureMode = Literal["peak", "volume"]
SCALAR_CALIB = {"peak": "scalar_calib.json", "volume": "scalar_calib_volume.json"}  # DIII-D fit per mode
LI_HALF = 0.55  # li / 2 at the DIII-D training mean li (1.10, scorer-derived from EFIT)


def core_pressure(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, Z, pe): each core Thomson point's position and its electron pressure [Pa] per EFIT frame, (T, n)."""
    row = read_row(path, exclude=LABEL_COLUMNS)
    pos, Te, ne, _ = thomson_geometry(row)
    pe = profile_on_efit(np.asarray(row["thomson_core_times"], float), ne * Te * EV, np.asarray(row["efit_times"]))
    return *core_chord_rz(row, np.asarray(pos, float)), pe


def peak_pressure(path: Path) -> np.ndarray:
    """(T,) electron pressure [Pa] per EFIT frame: 90th percentile over core Thomson chords of ne * Te."""
    pe = core_pressure(path)[2]
    pe = np.where(np.isfinite(pe) & (pe > 0), pe, np.nan)
    ok = np.isfinite(pe).any(1)
    out = np.full(len(pe), np.nan)
    out[ok] = np.nanpercentile(pe[ok], 90, axis=1)
    return out


def volume_pressure(path: Path, shape: np.ndarray, z0: np.ndarray | None = None) -> np.ndarray:
    """(T,) volume-averaged electron pressure [Pa] per EFIT frame on elliptical surfaces from shape rows
    (R0, a, kappa, ...) and centre heights z0 (the midplane by default)."""
    R, Z, pe = core_pressure(path)
    r0, a, kappa = (np.asarray(shape[:, k], float)[:, None] for k in range(3))
    zc = np.zeros_like(r0) if z0 is None else np.asarray(z0, float)[:, None]
    rho = np.sqrt(ellipse_rho2(R[None], Z[None], r0, a, zc, kappa))
    return np.array([volume_average(rho[t], pe[t]) for t in range(len(pe))])


def volume_average(rho: np.ndarray, p: np.ndarray) -> float:
    """integral p(rho) 2 rho d rho over [0, 1] from samples at radii `rho`, falling to 0 at the boundary."""
    ok = np.isfinite(p) & (p > 0) & (rho < 1)
    if ok.sum() < 3:
        return np.nan
    order = np.argsort(rho[ok])
    x, y = np.r_[rho[ok][order], 1.0], np.r_[p[ok][order], 0.0]
    f = np.interp(RHO, x, y) * 2 * RHO
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(RHO)))  # trapezoid rule


def raw_scalars(
    shape: np.ndarray,
    i_tf_ka: np.ndarray,
    ip_ka: np.ndarray,
    pe: np.ndarray,
    machine: str,
    eps_max: float = np.inf,
) -> np.ndarray:
    """(N, 2) uncalibrated q95 and betaN; shape rows are (R0, a, kappa, delta). NaN where an input is missing.
    eps_max caps the inverse aspect ratio in the q95 shaping factor, whose (1 - eps^2)^-2 diverges near eps = 1."""
    r0, a, kappa, delta = shape.T
    b0 = MU0 * TF_TURNS[machine] * np.abs(i_tf_ka) * 1e3 / (2 * np.pi * r0)
    ip = np.maximum(np.abs(ip_ka) / 1e3, MIN_IP_MA)
    eps = np.minimum(a / r0, eps_max)
    shaping = (1 + kappa**2 * (1 + 2 * delta**2 - 1.2 * delta**3)) / 2 * (1.17 - 0.65 * eps) / (1 - eps**2) ** 2
    q95 = 5 * a**2 * b0 / (r0 * ip) * shaping
    beta_n = 200 * MU0 * pe * a / (b0 * ip)  # beta_t[%] * a B0 / Ip with beta_t = 2 mu0 p / B0^2
    return np.stack([q95, beta_n], 1)


@dataclass(frozen=True)
class ScalarCalibration:
    c_q: float
    c_b: float
    q95_max: float
    fallback: tuple[float, float]  # DIII-D train means, used where an input is missing

    @classmethod
    def fit(cls, raw: np.ndarray, target: np.ndarray) -> ScalarCalibration:
        ok = np.isfinite(raw).all(1) & np.isfinite(target).all(1) & (raw > 0).all(1) & (target > 0).all(1)
        c = np.exp(np.median(np.log(target[ok] / raw[ok]), axis=0))
        q95_max = float(np.nanpercentile(target[:, 0], 99))
        return cls(float(c[0]), float(c[1]), q95_max, tuple(float(v) for v in np.nanmean(target, 0)))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        out = raw * np.array([self.c_q, self.c_b])
        out[:, 0] = np.minimum(out[:, 0], self.q95_max)
        return np.where(np.isfinite(out), out, np.array(self.fallback)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> ScalarCalibration:
        d = json.loads(path.read_text())
        return cls(**{**d, "fallback": tuple(d["fallback"])})


def tracking_stats(scal: np.ndarray, ref: np.ndarray, shot_index: np.ndarray) -> dict[str, float]:
    """Label-free comparison of (N, 2) q95 / betaN with a reference file: percentiles, correlation and mean |diff|
    with `ref`, and within-shot smoothness: the median step between consecutive frames and the von Neumann ratio
    (mean squared step / twice the within-shot variance: ~1 for white noise, ~0 for a smooth trace)."""
    same_shot = np.diff(shot_index) == 0
    inv = np.unique(shot_index, return_inverse=True)[1]
    out: dict[str, float] = {}
    for j, name in enumerate(("q95", "betaN")):
        v, r = np.asarray(scal[:, j], np.float64), np.asarray(ref[:, j], np.float64)
        step = np.abs(np.diff(v))[same_shot]
        within = v - (np.bincount(inv, v) / np.bincount(inv))[inv]
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        out |= {f"{name}_p5": p5, f"{name}_p50": p50, f"{name}_p95": p95}
        out |= {f"{name}_corr": float(np.corrcoef(v, r)[0, 1]), f"{name}_mad": float(np.mean(np.abs(v - r)))}
        out |= {f"{name}_step": float(np.median(step)), f"{name}_vn": float(np.mean(step**2) / (2 * np.var(within)))}
    return out


def shafranov_drive(r0: np.ndarray, a: np.ndarray, kappa: np.ndarray, pe: np.ndarray, ip_ka: np.ndarray) -> np.ndarray:
    """eps (beta_p + li / 2), what the large-aspect-ratio Shafranov shift Delta / a grows with, with li / 2 held at
    LI_HALF and beta_p = 2 mu0 <p> / <B_p>^2 from the Thomson electron pressure and the plasma current,
    <B_p> = mu0 Ip / (2 pi a sqrt((1 + kappa^2) / 2))."""
    ip = np.maximum(np.abs(ip_ka) / 1e3, MIN_IP_MA) * 1e6
    bp = MU0 * ip / (2 * np.pi * a * np.sqrt((1 + kappa**2) / 2))
    return a / r0 * (2 * MU0 * pe / bp**2 + LI_HALF)
