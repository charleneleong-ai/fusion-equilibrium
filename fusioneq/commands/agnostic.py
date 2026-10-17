"""The machine-agnostic network (docs/specs/2026-09-29-machine-agnostic-network-design.md): frame and pipeline exit
tests, training with a MAST sanity check, and the diagnosis that located its consistency ceiling."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from fusioneq.agnostic import FRAME, N_MOMENTS, MachineFrame, pipeline_plasma, vacuum_fields
from fusioneq.agnostic_model import AgnosticConfig, AgnosticPredictor, LossMode, MachineData, OffsetMode, machine_data
from fusioneq.cache import Frames
from fusioneq.evaluation import save_score
from fusioneq.greens import cached_grid_greens
from fusioneq.loading import load_predictor
from fusioneq.physics import (
    CHUNK,
    MAST_AXIS_SIGN,
    Calibration,
    MastCache,
    PhysicsEval,
    diii_d_geometry,
    ecoil_flux,
    frames_parts,
)
from fusioneq.plasma import PlasmaShape
from fusioneq.score import lcfs_shapes, summary_line
from fusioneq.settings import CACHE_ROOT, DATA_ROOT, DEVICE, RESULTS
from fusioneq.tracking import Run


def diii_d_data(ev: PhysicsEval, offset_features: bool) -> MachineData:
    """MachineData for the held-out DIII-D frames' cache, with the calibrated E-coil turns."""
    return machine_data(
        "DIII-D", ev.fr, ev.cache_root / "diii_d_train", ev.calib, DEVICE, ev.coil_file, offset_features
    )


def frame_check(cache_root: Path = CACHE_ROOT, n: int = 4000, seed: int = 0, size: int = FRAME) -> None:
    """M0 exit test: round-trip error of real psi / psi_vac through the machine-normalised frame, and channel ranges."""
    rng = np.random.default_rng(seed)
    for config, machine in [("diii_d_train", "DIII-D"), ("mast_public_test", "MAST")]:
        fr, frame = Frames.load(cache_root / config), MachineFrame.for_machine(machine, size)
        if not (np.allclose(frame.grid_R, fr.grid_R) and np.allclose(frame.grid_Z, fr.grid_Z)):
            raise ValueError(f"{machine}: the cache grid differs from the scorer mask's grid")
        rows = np.sort(rng.choice(len(fr.shot_index), n, replace=False))
        fields = {"psi_vac": fr.psi_vac} | ({"psi": fr.psi} if fr.psi is not None else {})
        env = torch.as_tensor(frame.envelope)
        for name, arr in fields.items():
            x = torch.as_tensor(np.asarray(arr[rows], np.float64))
            err = (frame.from_frame(frame.to_frame(x)) - x)[:, env].abs().amax(1)
            span = x[:, env].amax(1) - x[:, env].amin(1)
            rel = (err / span.clamp_min(1e-12)).numpy()
            logger.info(
                f"{machine} {name}: round-trip error / range p50 {np.median(rel):.2e}, p99 {np.percentile(rel, 99):.2e}"
            )
        ch = frame.channels(
            torch.as_tensor(np.asarray(fr.psi_vac[rows], np.float64)), torch.as_tensor(fr.ip[rows], dtype=torch.float64)
        )
        env = ch[:, 2] > 0.5
        v = ch[:, 0][env].numpy()
        logger.info(
            f"{machine} channels: vac/(mu0 Ip R_c) inside envelope p5/50/95 {np.percentile(v, [5, 50, 95]).round(3)}, "
            f"R/R_c range {ch[0, 1][env[0]].min():.2f}-{ch[0, 1][env[0]].max():.2f}"
        )


def pipeline_check(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    kappa: float = 1.6,
    alpha: float = 1.0,
) -> None:
    """M1 exit test: the ellipse baseline routed through the machine-normalised frame reproduces the machine-grid path
    (DIII-D physics-check scores, MAST psi_phys.npy)."""
    ev, shape = PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root), PlasmaShape(kappa, alpha)
    fr, rows, calib = ev.fr, ev.rows, ev.calib
    geom, ecoil = diii_d_geometry(fr, rows, ev.shot_dir), ecoil_flux(fr, rows, ev.coil_file)
    vac, old = frames_parts(fr, rows, geom, shape, ev.G)
    Gt = torch.as_tensor(ev.G.T, device=DEVICE)
    new = pipeline_plasma(MachineFrame.for_machine("DIII-D"), geom, fr.ip[rows], Gt, shape).reshape(old.shape)
    for label, plasma in [("machine grid", old), ("frame pipeline", new)]:
        for ec_label, ec in [("with E-coil", ecoil), ("no E-coil", 0.0)]:
            logger.info(f"DIII-D {label}, {ec_label}: {summary_line(ev.score(calib.apply(vac, plasma, ec)))}")
    mast_pipeline_check(cache_root, calib, shape)


def mast_pipeline_check(cache_root: Path, calib: Calibration, shape: PlasmaShape) -> None:
    """Rebuild every MAST frame through the frame pipeline and compare with psi_phys.npy, chunk by chunk."""
    mast = MastCache(cache_root)
    fr, geom, frame = mast.fr, np.load(mast.dir / "plasma_geom.npy"), MachineFrame.for_machine("MAST")
    Gt = torch.as_tensor(cached_grid_greens(mast.dir, fr.grid_R, fr.grid_Z).T, device=DEVICE)
    stored = np.load(mast.dir / "psi_phys.npy", mmap_mode="r")
    rel, ss_res, total, total_sq, count = [], 0.0, 0.0, 0.0, 0
    for i in range(0, len(geom), CHUNK):
        sl = slice(i, i + CHUNK)
        old = np.asarray(stored[sl], np.float32).reshape(len(geom[sl]), -1)
        vac = np.asarray(fr.psi_vac[sl], np.float32).reshape(len(old), -1)
        new = calib.apply(vac, pipeline_plasma(frame, geom[sl], fr.ip[sl], Gt, shape))
        rel.append(np.abs(new - old).max(1) / np.ptp(old, 1))
        ss_res += float(np.sum((new - old) ** 2))
        total, total_sq, count = total + float(old.sum()), total_sq + float(np.sum(old**2)), count + old.size
    r2 = 1 - ss_res / (total_sq - total**2 / count)
    p50, p99 = np.percentile(np.concatenate(rel), [50, 99])
    logger.info(f"MAST frame pipeline vs psi_phys.npy: R2 {r2:.6f}, max error / range p50 {p50:.2e} p99 {p99:.2e}")


def train_agnostic(
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
    tag: str = "agnostic",
    epochs: int = 8,
    frames_per_epoch: int = 300_000,
    batch: int = 64,
    lr: float = 1e-3,
    base: int = 32,
    seed: int = 0,
    loss: LossMode = "shape",
    offset: OffsetMode = "vacuum",
    correction: bool = False,
    correction_weight: float = 0.0,
    scalars_from: str = "v5_loss_mix_i03",
) -> None:
    """M2: train the current-predicting U-Net on DIII-D, score it on the held-out shots (q95/betaN borrowed from
    `scalars_from` so S is comparable), then predict MAST (psi_<tag>.npy) and run the input-only sanity checks."""
    t0, ev = time.time(), PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root)
    fr, calib = ev.fr, ev.calib
    d3d = diii_d_data(ev, offset == "vacuum")
    cfg = AgnosticConfig(
        base=base,
        epochs=epochs,
        frames_per_epoch=frames_per_epoch,
        batch=batch,
        lr=lr,
        seed=seed,
        loss=loss,
        offset=offset,
        correction=correction,
        correction_weight=correction_weight,
    )
    run = Run(tag, {"model": "agnostic", **asdict(cfg), "holdout": holdout}, job_type="agnostic")
    model = AgnosticPredictor(cfg, calib, DEVICE).fit(fr, ev.train, d3d)
    model.save(RESULTS / tag)
    psi = model.predict(fr, ev.rows, d3d)
    _, scal = load_predictor(RESULTS / scalars_from).predict(fr, ev.subset)
    res = ev.score(psi, scal)
    logger.info(f"{tag} DIII-D: {summary_line(res)}")
    mast = agnostic_mast_check(model, cache_root, calib, tag)
    record = {"model": "agnostic", **asdict(cfg), "scalars_from": scalars_from, "mast_check": mast}
    save_score(tag, record, res, run, t0)
    run.finish()


def agnostic_mast_check(
    model: AgnosticPredictor, cache_root: Path, calib: Calibration, tag: str, n: int = 400
) -> dict[str, float]:
    """Predict every MAST frame (psi_<tag>.npy) and check the boundaries the official extractor finds on a sample."""
    mast = MastCache(cache_root)
    fr = mast.fr
    data = machine_data("MAST", fr, mast.dir, calib, DEVICE, offset_features=model.cfg.offset == "vacuum")
    psi = model.predict(fr, np.arange(len(fr.shot_index)), data)
    np.save(mast.dir / f"psi_{tag}.npy", psi.astype(np.float16))
    rows = np.random.default_rng(0).choice(len(psi), n, replace=False)
    shapes = lcfs_shapes(psi[rows], fr.grid_R, fr.grid_Z, "MAST", MAST_AXIS_SIGN)  # no envelope (rule 3)
    out = {
        "closed_frac": float(np.isfinite(shapes).all(1).mean()),
        "r_geo_median": float(np.nanmedian(shapes[:, 0])),
        "kappa_median": float(np.nanmedian(shapes[:, 2])),
    }
    logger.info(f"MAST agnostic: {out}")
    return out


def envelope_mean(field: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """(T, 1, 1) per-frame mean of a (T, nz, nr) field inside the envelope."""
    return field[:, envelope].mean(1)[:, None, None]


@dataclass
class AgnosticDiagnosis:
    """Oracle variants of a trained model's DIII-D prediction: its offset and shape errors apart, and the labels' own
    current through the same vacuum + Green's step, with label-fitted offsets or vacuum-field corrections."""

    ev: PhysicsEval
    data: MachineData

    def vacuum_fit(self, resid: np.ndarray) -> np.ndarray:
        """(T, nz, nr) least-squares fit of the six vacuum fields to a residual inside the envelope, evaluated
        everywhere: the best external-field correction any per-frame head could add."""
        frame = self.data.frame
        R, Z = np.meshgrid(frame.grid_R, frame.grid_Z)
        r, z = R / frame.r_c, Z / frame.r_c
        coef = resid.reshape(len(resid), -1) @ MachineFrame.moment_pinv(r, z, frame.envelope).T
        return (coef @ vacuum_fields(r, z).reshape(N_MOMENTS, -1)).reshape(resid.shape).astype(np.float32)

    @torch.no_grad()
    def label_current_psi(self, label: np.ndarray, chunk: int = 512) -> np.ndarray:
        """(T, nz, nr) psi from the calibrated vacuum plus the flux of each label's own current at the measured Ip."""
        data, rows, out = self.data, self.ev.rows, []
        for i in range(0, len(rows), chunk):
            r = rows[i : i + chunk]
            target = torch.as_tensor(label[i : i + chunk], device=data.Gt.device).flatten(1)
            ip_a = data.ip_ka[torch.as_tensor(r, device=data.Gt.device)].abs() * 1e3
            J = data.frame.label_current(target) * ip_a[:, None]
            out.append(self.ev.calib.apply(data.vacuum(self.ev.fr, r).flatten(1), J @ data.Gt).cpu().numpy())
        return np.concatenate(out).reshape(label.shape)

    def variants(self, tag: str, pred: np.ndarray) -> dict[str, np.ndarray]:
        label = np.asarray(self.ev.fr.psi[self.ev.rows], np.float32)
        true_current = self.label_current_psi(label)
        env = self.data.frame.envelope
        offset_err = envelope_mean(pred - label, env)
        return {
            tag: pred,
            "offset error only": label + offset_err,
            "shape error only": pred - offset_err,
            "label current": true_current,
            "label current + label offset": true_current + envelope_mean(label - true_current, env),
            "label current + label vacuum fit": true_current + self.vacuum_fit(label - true_current),
            f"{tag} + label vacuum fit": pred + self.vacuum_fit(label - pred),
        }


def agnostic_diagnose(
    tag: str = "agnostic_m2c_i00",
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
    holdout: int = 200,
    score_shots: int = 40,
) -> None:
    """Score AgnosticDiagnosis's variants of a trained agnostic model. Scalars are zero throughout: compare R2psi,
    D_LCFS and consistency."""
    ev = PhysicsEval.load(cache_root, holdout, score_shots, data_root=data_root)
    model = AgnosticPredictor.load(RESULTS / tag, DEVICE)
    data = diii_d_data(ev, model.cfg.offset == "vacuum")
    out = {}
    for name, psi in AgnosticDiagnosis(ev, data).variants(tag, model.predict(ev.fr, ev.rows, data)).items():
        res = ev.score(psi)
        logger.info(f"{name}: {summary_line(res)}")
        out[name] = {k: res[k] for k in ("r2_psi", "dlcfs", "consistency", "r2_cons_each")}
    (RESULTS / f"{tag}_diagnose.json").write_text(json.dumps(out, indent=1))


COMMANDS = [frame_check, pipeline_check, train_agnostic, agnostic_diagnose]
