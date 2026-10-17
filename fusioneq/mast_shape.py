"""Per-frame plasma shape for the physics flux (lever 1b) and the label-free checks that keep it safe.

The plasma ellipse's centre height comes from the DIII-D-fitted ShapeMap (the scorer's O-point follows it one to one);
its elongation is picked per frame from a short ladder so that the scorer's derived kappa of our own flux lands nearest
the map's kappa. Current-density triangularity does not move the derived triangularity (the X-points sit in the coil
field), so triangularity is predicted but not imposed.

Every candidate is derived by the scorer's own code on our flux alone - no labels, no envelope mask - and must pass the
gate: a closed LCFS, the O-point inside the plasma ellipse and derived kappa within KAPPA_BAND. A frame where no shaped
candidate passes keeps the fixed-shape flux. The O-point may not sit inside a coil conductor: where a coil's own flux
outruns the plasma's (MAST's P2 coils on ~3% of frames), the flux inside that coil is capped just below the plasma
peak.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from fusioneq.score import derive_frame, extract_lcfs_with_sign

DERIVED = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li", "lcfs")
KAPPA_LADDER = (1.0, 1.3, 1.6, 2.0, 2.4)
AXIS_STEP = 0.1  # the most one O-point feedback step moves the ellipse centre [m]
SERVED = np.float16  # the dtype the flux is served in: every gate judges the rounded flux
FIXED_KAPPA = 1.6  # the fixed ellipse of the current entry, Z0 = 0
# Sane derived elongation, set from DIII-D and physics, never from MAST labels: DIII-D's scorer-derived kappa over 386k
# training frames spans 1.00-2.20 (p99.9 2.05); the floor is that minimum, and the ceiling adds 0.3 of headroom for a
# low-aspect-ratio machine's higher natural elongation. MAST's fixed-shape tail (p95 3.5) is far outside either.
KAPPA_BAND = (1.0, 2.5)
AXIS_RADIUS = 0.8  # the O-point must sit within this normalised radius of the plasma ellipse
# (kappa_e, place rows (R0, a, Z0), optional frame mask) -> (N, nz, nr) plasma flux of those frames
PlasmaFlux = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Deriver:
    """The scorer's seven scalars of our own flux plus 1/0 for a closed LCFS, per frame. `masks` (mask_coarse, mask_f)
    is for DIII-D only: on MAST the scorer's envelope may not touch predictions (rule 3), so it stays None."""

    R: np.ndarray
    Z: np.ndarray
    machine: str
    axis_sign: float
    masks: tuple[np.ndarray, np.ndarray] | None = None
    with_li: bool = False

    def rows(self, psi: np.ndarray) -> np.ndarray:
        """(N, len(DERIVED))."""
        mc, mf = self.masks or (None, None)
        out = np.full((len(psi), len(DERIVED)), np.nan)
        for k, p in enumerate(np.asarray(psi, np.float64)):
            c, _ = extract_lcfs_with_sign(p, self.R, self.Z, self.machine, mc, mf, axis_sign=self.axis_sign)
            v = derive_frame(p, self.R, self.Z, self.machine, mc, mf, contour=c, axis_sign=self.axis_sign,
                             with_li=self.with_li)  # fmt: skip
            out[k] = [*(v[n] for n in DERIVED[:-1]), c is not None]
        return out

    def many(self, psi: np.ndarray, workers: int) -> np.ndarray:
        chunks = np.array_split(np.arange(len(psi)), max(1, min(len(psi), workers * 4)))
        with ProcessPoolExecutor(workers) as ex:
            return np.concatenate(list(ex.map(self.rows, [np.asarray(psi[c]) for c in chunks])))


def coil_guard(grid_R: np.ndarray, grid_Z: np.ndarray, coils: dict[str, np.ndarray]) -> np.ndarray:
    """(nz, nr) grid cells inside any coil element's rectangle widened by one grid cell."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    dr, dz = grid_R[1] - grid_R[0], grid_Z[1] - grid_Z[0]
    out = np.zeros(RR.shape, bool)
    for r, z, w, h in zip(
        *(np.asarray(coils[k], float) for k in ("coil_R", "coil_Z", "coil_width", "coil_height")), strict=True
    ):
        out |= (np.abs(RR - r) <= w / 2 + dr) & (np.abs(ZZ - z) <= h / 2 + dz)
    return out


def plasma_region(grid_R: np.ndarray, grid_Z: np.ndarray, geom: np.ndarray, kappa: float) -> np.ndarray:
    """(N, nz, nr) cells inside each frame's plasma ellipse; geom rows (R0, a, Z0)."""
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    r0, a, z0 = (geom[:, k, None, None] for k in range(3))
    return ((RR - r0) / a) ** 2 + ((ZZ - z0) / (kappa * a)) ** 2 <= 1


def cap_coils(psi: np.ndarray, guard: np.ndarray, region: np.ndarray, axis_sign: float) -> np.ndarray:
    """psi with phi = axis_sign * psi held just below each frame's plasma peak inside the coil guard."""
    phi = axis_sign * np.asarray(psi, np.float64)
    peak = np.where(region & ~guard, phi, -np.inf).max((1, 2))
    cap = peak - 5e-3 * np.ptp(phi, axis=(1, 2))  # a margin float16 rounding cannot close
    capped = np.where(guard & (phi > cap[:, None, None]), cap[:, None, None], phi)
    return (axis_sign * capped).astype(np.float32)


def axis_inside(derived: np.ndarray, geom: np.ndarray, kappa: float) -> np.ndarray:
    rho2 = ((derived[:, 0] - geom[:, 0]) / geom[:, 1]) ** 2 + ((derived[:, 1] - geom[:, 2]) / (kappa * geom[:, 1])) ** 2
    return rho2 <= AXIS_RADIUS**2


def gate(derived: np.ndarray, geom: np.ndarray, kappa: float, band: tuple[float, float] = KAPPA_BAND) -> np.ndarray:
    """(N,) closed LCFS, O-point inside the plasma, derived kappa in the sane band."""
    k = derived[:, 2]
    return (derived[:, 7] == 1) & np.isfinite(k) & (k >= band[0]) & (k <= band[1]) & axis_inside(derived, geom, kappa)


def choose(kappa: np.ndarray, ok: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,) index of the passing candidate whose derived kappa (C, N) is nearest the target, -1 where none passes."""
    err = np.where(ok, np.abs(kappa - target[None]), np.inf)
    best = err.argmin(0)
    return np.where(np.isfinite(err.min(0)), best, -1)


@dataclass
class ShapeLadder:
    """Builds and gates the candidates for a block of frames: the fixed shape first, then the ladder at the map's Z.
    A candidate is an elongation and place rows (R0, a, Z0) of the plasma ellipse."""

    deriver: Deriver  # unmasked
    guard: np.ndarray
    workers: int = 8
    ladder: tuple[float, ...] = KAPPA_LADDER

    def candidate(self, base: np.ndarray, plasma: PlasmaFlux, place: np.ndarray, kappa: float) -> np.ndarray:
        d = self.deriver
        region = plasma_region(d.R, d.Z, place, kappa)
        psi = cap_coils(base + plasma(kappa, place), self.guard, region, d.axis_sign)
        return psi.astype(SERVED).astype(np.float32)

    def judge(self, psi: np.ndarray, place: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
        d = self.deriver.many(psi, self.workers)
        return d, gate(d, place, kappa)

    def refine(
        self, base: np.ndarray, plasma: PlasmaFlux, rec: dict[str, np.ndarray], target: np.ndarray, axis: int
    ) -> np.ndarray:
        """One feedback step on the ellipse centre so the derived O-point (axis 0: R, 1: Z) lands on `target`; kept
        per frame only where the gate still passes and the O-point moved closer. Updates `rec`; returns the new flux
        rows for frames it changed (NaN elsewhere)."""
        out = np.full(base.shape, np.nan, np.float32)
        for kappa in np.unique(rec["kappa_e"][rec["pick"] > 0]):
            m = (rec["pick"] > 0) & (rec["kappa_e"] == kappa)
            err = target[m] - rec["derived"][m, axis]
            place = rec["place"][m].copy()
            place[:, 2 * axis] += np.clip(err, -AXIS_STEP, AXIS_STEP)  # place column 0 is R0, column 2 is Z0
            sub = lambda k, p, m=m: plasma(k, p, m)  # noqa: E731 - the flux of the selected frames only
            psi = self.candidate(base[m], sub, place, float(kappa))
            d, ok = self.judge(psi, place, float(kappa))
            take = ok & (np.abs(target[m] - d[:, axis]) < np.abs(err))
            idx = np.flatnonzero(m)[take]
            out[idx], rec["place"][idx], rec["derived"][idx] = psi[take], place[take], d[take]
        return out

    def run(
        self,
        base: np.ndarray,
        plasma: PlasmaFlux,
        geom: np.ndarray,
        z_target: np.ndarray,
        kappa_target: np.ndarray,
        r_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """(psi, record) for frames with base flux `base` and Thomson / label geometry rows (R0, a); the O-point is fed
        back onto z_target and then, if given, r_target (a Shafranov-shifted axis)."""
        n = len(base)
        fixed, shaped = np.c_[geom[:, :2], np.zeros(n)], np.c_[geom[:, :2], z_target]
        shapes = [(FIXED_KAPPA, fixed)] + [(k, shaped) for k in self.ladder]
        psis, derived, ok = [], [], []
        for kappa, place in shapes:
            psi = self.candidate(base, plasma, place, kappa)
            d, passed = self.judge(psi, place, kappa)
            psis.append(psi)
            derived.append(d)
            ok.append(passed)
        derived_a, ok_a = np.stack(derived), np.stack(ok)
        pick = choose(derived_a[1:, :, 2], ok_a[1:], kappa_target) + 1  # 0 = keep the fixed shape
        psi = np.stack(psis)[pick, np.arange(n)]
        record = {
            "pick": pick,
            "kappa_e": np.array([s[0] for s in shapes])[pick],
            "place": np.where((pick > 0)[:, None], shaped, fixed),
            "fixed_ok": ok_a[0],
            "any_ok": ok_a.any(0),
            "derived_fixed": derived_a[0],
            "derived": derived_a[pick, np.arange(n)],
        }
        for name, axis, target in (("z_refined", 1, z_target), ("r_refined", 0, r_target)):
            if target is None:
                continue
            refined = self.refine(base, plasma, record, target, axis)
            changed = np.isfinite(refined[:, 0, 0])
            psi[changed] = refined[changed]
            record[name] = changed
        record["z0"] = record["place"][:, 2]
        return psi, record


def summary(derived: np.ndarray, shot: np.ndarray, geom: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Label-free table of derived scalars: per scalar the NaN share, p5/p50/p95, the median within-shot step between
    consecutive frames and the within-shot share of variance; plus LCFS failures, kappa outside KAPPA_BAND and, with
    geom rows (R0, a, Z0), the O-point outside the plasma."""
    out: dict[str, dict[str, float]] = {}
    same = shot[1:] == shot[:-1]
    for j, name in enumerate(DERIVED[:-1]):
        v = derived[:, j]
        ok = np.isfinite(v)
        if not ok.any():
            out[name] = {"nan": 1.0}
            continue
        p5, p50, p95 = np.percentile(v[ok], [5, 50, 95])
        step = np.abs(np.diff(v))[same]
        shot_mean = {s: np.nanmean(v[shot == s]) for s in np.unique(shot[ok])}
        within = np.nanvar(v - np.array([shot_mean.get(s, np.nan) for s in shot]))
        out[name] = {
            "nan": float(1 - ok.mean()),
            "p5": float(p5),
            "p50": float(p50),
            "p95": float(p95),
            "step": float(np.nanmedian(step)) if np.isfinite(step).any() else np.nan,
            "within": float(within / max(np.nanvar(v), 1e-30)),
        }
    k = derived[:, 2]
    flags = {"lcfs_fail": float(np.mean(derived[:, 7] != 1)),
             "kappa_out": float(np.mean(np.isfinite(k) & ((k < KAPPA_BAND[0]) | (k > KAPPA_BAND[1]))))}  # fmt: skip
    if geom is not None:
        flags["axis_off"] = float(np.mean(~axis_inside(derived, geom, FIXED_KAPPA)))
    out["flags"] = flags
    return out
