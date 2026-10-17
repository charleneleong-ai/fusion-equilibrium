# MAST per-frame plasma shape from the vacuum field (lever 1b), 2026-10-11

## Hypothesis

MAST consistency (0.0117 on `4c428222`) is low not because derivation fails (0% LCFS failures on our flux) but
because the derived shape scalars do not track the truth. The fixed ellipse holds kappa at 1.6 and Z0 at 0. A
dimensionless map from vacuum-field moments, fitted on DIII-D, should let the plasma's elongation and centre height
follow the coils frame by frame. A label-free gate should trim the outlier tail that zeroes clipped R2 terms.

## Rule 3 compliance

No MAST or MAST-U equilibrium data enters any step that produces MAST predictions. That covers EFIT psi, the demo
shots' labels, the scorer's `mast_envelope.npz` and probe submissions.

- The map is fitted and validated on DIII-D only. The training targets are the scorer-derived scalars of DIII-D EFIT
  flux (`cons_targets.npz`), and its geometry comes from the DIII-D EFIT boundary.
- On MAST it reads coil currents, Ip, coil geometry and the Thomson-placed (R0, a) already in `plasma_geom.npy`.
- The ladder selection and the gate run the scorer's derivation on **our own** flux, with no labels and no envelope
  mask ([`Deriver`](../../../fusioneq/mast_shape.py)).
- `SolenoidOffset` and `DemoScalarFit` are not used. The kappa band comes from DIII-D and physics (below).
- Nothing was submitted or pushed.

## Method

1. **Map** ([`fusioneq/shape_map.py`](../../../fusioneq/shape_map.py)). The raw vacuum flux is fitted with a
   least-squares cubic about (R0, 0), in units of a, sampled on a fixed elliptical stencil. On DIII-D the vacuum flux
   includes the missing E-coil turns; on MAST the solenoid is at half current, the shape the current entry uses. Eight
   features are kept, all ratios and so invariant under psi -> -psi:
   - field-line tilt c01/c10;
   - decay index n = 1 - 2 (R0/a) c20/c10;
   - quadrupole tilt c11/c10;
   - four hexapole ratios;
   - Shafranov's Lambda from |Bz| over the hoop field mu0 Ip / 4 pi R0.

   The model is one ridge per target (Z_axis, kappa, tri_top, tri_bot), with features clipped to DIII-D's p0.5-p99.5
   and predictions to the DIII-D target range. Lambda was chosen by shot-grouped 3-fold CV (picked 1e-4, the smallest
   tried). `shape-fit` fits on 45,713 frames of 800 train shots.
2. **Realisation** ([`ShapeLadder`](../../../fusioneq/mast_shape.py)). Per frame, the candidates are:
   - the fixed shape (kappa_e 1.6, Z0 0);
   - kappa_e in {1.0, 1.3, 1.6, 2.0, 2.4} with Z0 = the map's Z_axis.

   Each candidate is derived by the scorer's code with no mask, after float16 rounding (the served dtype). The gate
   needs a closed LCFS, the O-point within 0.8 of the plasma ellipse, and derived kappa in [1.0, 2.5]. Among the
   passing candidates, the one whose derived kappa is nearest the map's kappa wins. One feedback step then moves Z0 by
   the O-point's miss (at most 0.1 m), and is kept only where the gate still passes and the miss shrinks. A frame with
   no passing candidate keeps the fixed shape.
3. **Coil guard** (geometry only). On ~2-4% of MAST frames the scorer's O-point (the strongest interior local maximum)
   sits inside a P2 coil at (0.51, ±1.74) m, where the coil's own flux outruns the plasma's. Inside the shipped coil
   rectangles, widened by one cell, the flux is capped 0.5% of the frame's range below the plasma peak.

   The v1 build used a cap margin of 1e-6. Float16 rounding erased it: the served file showed 1.3% LCFS failures and
   5.7% off-plasma axes, while the gate had seen 0% and 1.5%. v2 judges the rounded flux.

Commands: `shape-fit`, `shape-check`, `mast-shape`, `derive-check` in
[`fusioneq/commands/shape.py`](../../../fusioneq/commands/shape.py). `derive-check` replaces the ad-hoc
`derive_check.py`. Tests: [`tests/test_shape.py`](../../../tests/test_shape.py).

**Kappa band [1.0, 2.5].** DIII-D's scorer-derived kappa over 386,898 frames spans 1.00-2.20 (p99.9 2.05). The floor
is that minimum. The ceiling adds 0.3 of headroom for a low-aspect-ratio machine's higher natural elongation. This is
a judgement, not a measurement. No MAST label was consulted.

## DIII-D held-out results (40 scoring shots, 8,222 frames)

The map alone, as R2 against the scorer-derived truth:

| target | map R2 | constant (train mean) |
|---|---|---|
| Z_axis | **+0.534** | -0.066 |
| kappa | **+0.425** | -0.000 |
| tri_top | **+0.687** | -0.024 |
| tri_bot | **+0.445** | -0.003 |

The whole MAST procedure, replayed on the same shots (`shape-check`; R0 and a from labels in place of Thomson, scored
officially):

| DIII-D replay | S | R2psi | D_LCFS | cons | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed ellipse | 0.5089 | 0.7808 | 0.2054 | 0 | -7.64 | -10.55 | -8.41 | -0.68 | -0.97 | -12.4 | -0.44 |
| per-frame shape (v2) | **0.5126** | 0.7817 | **0.1732** | 0 | -8.01 | -9.22 | -3.48 | -0.56 | -0.54 | -12.1 | -0.42 |

The map works, but realising it in the physics superposition does not carry it into the scorer's derivation:

- Under the scorer's masked derivation, the shaped flux's Z_axis is off by a median 1.5 cm against a truth spread of
  3.7 cm, but the p95 error is 5.6 cm.
- kappa's median error is 0.08, but its p95 error is 0.58.

These tails, together with level biases (R_axis -2.4 cm, volume -3.0 m^3), keep every consistency term below 0. The
measured gain is S +0.0037, from D_LCFS (0.205 -> 0.173) and smaller kappa and triangularity penalties. Pick shares
(fixed, then 1.0 ... 2.4): 5 / 11 / 30 / 31 / 8 / 15%. The Z feedback was kept on 73% of frames.

## MAST label-free diagnostics (public test, every 10th frame, 6,908 frames, no mask)

Before is the current entry `4c428222` (`psi_k1.6_a2.0_f0.4.npy` minus the solenoid-shape correction, which
reproduces its served flux to 6e-5). After is the v2 candidate `psi_shape_dmap.npy`, read as served (float16). Each
cell is p5 / p50 / p95; "step" is the median change between consecutive sampled frames within a shot; "within" is
the share of variance that lies within shots.

| scalar | before p5 / p50 / p95 | after p5 / p50 / p95 | step before -> after | within before -> after |
|---|---|---|---|---|
| R_axis [m] | 0.739 / 0.894 / 0.964 | 0.781 / 0.894 / 0.961 | 0.024 -> 0.024 | 0.68 -> 0.74 |
| Z_axis [cm] | -0.14 / -0.01 / +0.43 | -1.50 / -0.94 / 0.00 | 0.015 -> 0.137 | 0.65 -> 0.78 |
| kappa | 1.466 / 1.765 / 2.008 | 1.665 / 1.799 / 1.927 | 0.056 -> 0.046 | 0.75 -> 0.83 |
| tri_top | 0.255 / 0.413 / 0.497 | 0.276 / 0.373 / 0.453 | 0.030 -> 0.024 | 0.67 -> 0.78 |
| tri_bot | 0.219 / 0.402 / 0.488 | 0.293 / 0.413 / 0.495 | 0.029 -> 0.028 | 0.73 -> 0.79 |
| volume [m^3] | 4.77 / 7.64 / 10.17 | 5.72 / 7.59 / 10.07 | 0.53 -> 0.52 | 0.67 -> 0.70 |
| li | 0.775 / 1.441 / 2.767 | 0.788 / 1.358 / 2.175 | 0.24 -> 0.18 | 0.77 -> 0.77 |

| flag | before | after |
|---|---|---|
| LCFS not closed | 0.0% | 0.0% |
| scorer's O-point outside the plasma (in a P2 coil) | 3.95% | **1.24%** |
| derived kappa outside [1.0, 2.5] | 4.11% | 3.79% |
| li not derivable | 1.10% | 1.17% |

v1 (cap margin erased by float16) served 1.29% LCFS failures and 5.67% off-plasma axes, worse than before; it is
superseded. The earlier first/mid/last-frame sample (726 frames) showed a kappa p95 of 3.5. Every-10th sampling
gives 2.0, so that tail sits mostly in ramp frames.

The candidate's pick shares and gate outcome across all 69,075 frames: fixed kept 4.8% (no shaped candidate passes), kappa_e 1.0 / 1.3 / 1.6 / 2.0 / 2.4 picked on 2.7 / 4.1 / 49.8 / 25.6 / 13.0%.

## Candidate files (on pi-a100-80gb)

- `~/fusion/cache/mast_public_test/psi_shape_dmap.npy`: float16 (69,075, 65, 65). Serve with
  `build --mast-psi psi_shape_dmap.npy --mast-scal scal_vol_eps065_a2f04.npy`; the q95 / betaN are unchanged from the
  current entry.
- `~/fusion/cache/mast_public_test/psi_shape_dmap_record.npz`: per-frame pick, kappa_e, z0, map targets, and derived
  scalars before and after.
- `~/fusion/submission_mast_shape_dmap/mast_public_test.npz`: built and validated (5 shots), MAST only.
- `~/fusion/cache/shape_map.json`: the DIII-D-fitted map. `~/fusion/cache/shape_check.npz`: the DIII-D replay arrays.
- Logs: `~/fusion/logs/mast_shape_fit_20261011T084207Z.log` (fit and v1 replay),
  `mast_shape_build_20261011T084546Z.log` (v1 MAST), `mast_shape_v2_20261011T100335Z.log` (v2 replay and MAST), and
  `mast_shape_derive_check.json`.

## What is not established

- **Whether MAST consistency moves at all.** On DIII-D, where it can be measured, the same procedure leaves
  consistency at 0. The MAST effect is known only through an official Development submission.
- **Masked versus unmasked derivation.** The gate uses no mask (rule 3). The scorer uses its envelope. On DIII-D the
  masked derivation produces the tails above, and the MAST equivalent is unmeasurable label-free.
- **The map's level on MAST.** It predicts MAST Z_axis around -0.9 cm (DIII-D's intercept and coefficients) and
  triangularity it cannot impose: the current's own triangularity does not move the derived value, because the
  X-points sit in the coil field. Whether DIII-D's intercepts suit MAST is untested.
- **The kappa band ceiling of 2.5** is a judgement.
- **Mask-free q95 / betaN against the score.** Their effect on the official q95+betaN term is unmeasured (next
  section).

## Rule-3 fix: mask-free MAST scalars

The MAST q95 / betaN (`scal_vol_eps065_a2f04.npy`, built by
[`MastCache.scalars`](../../../fusioneq/physics.py)) took our LCFS from `Scorer("MAST").boundary_shapes`. That loaded
the scorer's `mast_envelope.npz`, so the envelope shaped a MAST prediction. The current MAST entry carries those
scalars.

Every MAST prediction path is now mask-free:

- [`MastCache.boundary_shapes`](../../../fusioneq/physics.py) and the agnostic MAST check call
  [`lcfs_shapes`](../../../fusioneq/score.py) with no envelope.
- The closed agnostic model's MAST frame ([`MachineFrame.for_machine`](../../../fusioneq/agnostic.py)) is a geometry
  box: R from 0.2 to 1.9 m, which clears the centre-column solenoid (R <= 0.16 m), and |Z| <= 1.8 m.
- `Scorer("MAST")` remains only where we score against labels (`mast-check` on the demo shots), which is validation
  rather than prediction.
- [`TestRule3`](../../../tests/test_physics.py) makes `np.load` raise on `mast_envelope`, then builds the MAST
  boundary shapes on a synthetic cache and the agnostic MAST frame.

Without the envelope, the extractor sometimes closes around a P2 coil or around the field null instead of the plasma.
So [`plausible_boundary`](../../../fusioneq/physics.py) drops any boundary with:

- kappa outside [1.0, 2.5];
- a centre more than 0.5 a from the Thomson centre;
- a minor radius outside 0.5-1.5 times the Thomson a.

Dropped frames take the median boundary (`mast-scalars --gate`, the default). Filling them from the scaled Thomson
extent instead tracked worse (q95 correlation with the masked file 0.46), so it was not kept.

Label-free comparison with the masked file on all 69,075 public-test frames, from the same flux
(`psi_k1.6_a2.0_f0.4.npy`, eps cap 0.65, volume pressure):

| | masked (current) | mask-free, ungated | mask-free, gated |
|---|---|---|---|
| boundaries implausible (median-filled) | - | - | 30.6% |
| q95 p5 / p50 / p95 | 3.44 / 5.29 / 9.36 | 0.34 / 5.07 / 11.56 | 3.48 / 5.47 / 9.45 |
| q95 at its cap | ~1% | 8.8% | 2.4% |
| q95 frames changed (> 1e-6) | - | 22.8% | 32.6% |
| q95 max / mean abs diff | - | 10.74 / 0.89 | 11.18 / 0.55 |
| q95 correlation with masked | - | 0.775 | 0.778 |
| betaN p5 / p50 / p95 | 0.385 / 1.271 / 3.068 | 0.048 / 1.129 / 2.938 | 0.429 / 1.269 / 3.062 |
| betaN max / mean abs diff | - | 4.60 / 0.194 | 4.52 / 0.032 |
| betaN correlation with masked | - | 0.774 | 0.994 |

The new files are `~/fusion/cache/mast_public_test/scal_vol_eps065_a2f04_nomask_gated.npy` (the one to serve) and
`scal_vol_eps065_a2f04_nomask.npy` (ungated, for reference). The existing masked file is untouched. Logs:
`~/fusion/logs/mast_scal_nomask_20261011T111750Z.log` and `mast_scal_nomask_gated_20261011T115826Z.log`.

Not established: how the official q95+betaN term (0.320 now) moves. The gated file matches the masked
distributions, but a third of its frames take the median boundary, so frame-to-frame q95 tracking may drop. Serving
it once beside the current flux would measure that.

## Geometry-only extraction region and per-shot fill for MAST q95 / betaN, 2026-10-12

The mask-free gated scalars (`22d99887`) cost the official q95+betaN R2 0.320 -> 0.208. About a third of their
frames took the global median boundary. This section bounds the extractor with machine geometry instead of the
scorer's envelope, and fills dropped frames from their own shot. Rule 3 still holds: the region uses only the shipped
coil rectangles and the agnostic box, the DIII-D check uses DIII-D labels, and nothing was submitted or pushed.

**Region** ([`fusioneq/plasma_region.py`](../../../fusioneq/plasma_region.py)). Neither the shipped MAST inputs nor
FreeGS `MAST()` carry a limiter or vessel outline. FreeGS `MASTU()` does, but that is a different machine. So:

- MAST `chamber`: the agnostic box (R 0.2-1.9 m, |Z| <= 1.8 m) minus the shipped PF coil rectangles (dilated one
  cell; MAST's P2-P6 sit inside the vessel), kept only in line of sight of (1.05, 0) m;
- MAST `box`: the agnostic box alone;
- DIII-D `wall`: the first wall of FreeGS `DIIID()` (freegs-plasma/freegs master 5b41fe5).

The extractor gets the region as its mask, exactly where it took the envelope. Commands:

- `mast-scalars --region chamber --fill shot`;
- `mast-scal-compare` (label-free table);
- `region-check` (DIII-D replay).

**Fill** ([`fill_from_shot`](../../../fusioneq/physics.py)). A dropped frame is interpolated over the frame index from
its own shot's plausible frames, held at the shot's ends. A shot with none takes its own Thomson (R0, a) with the
median plausible (kappa, delta). That shot-less case hit 0.02% of frames.

### MAST label-free comparison (all 69,075 public-test frames, eps cap 0.65, volume pressure)

The reference is the masked file `scal_vol_eps065_a2f04.npy` (official 0.320). Correlation and mean |diff| are
against it. "step" is the median |change| between consecutive frames in a shot. "vN" is the von Neumann ratio:
mean squared step over twice the within-shot variance, so lower is smoother.

| file | flux | region | fill | dropped | q95 p5 / p50 / p95 | q95 corr | q95 mean abs diff | q95 step / vN | betaN corr | betaN vN |
|---|---|---|---|---|---|---|---|---|---|---|
| masked (served, rule-3 non-compliant) | a2f04 | envelope | - | - | 3.44 / 5.29 / 9.36 | 1 | 0 | 0.139 / 0.036 | 1 | 0.081 |
| `22d99887` nomask_gated | a2f04 | none | median | 30.6% | 3.48 / 5.47 / 9.45 | 0.778 | 0.553 | 0.125 / 0.131 | 0.994 | 0.082 |
| a2f04_none_shot | a2f04 | none | shot | 30.6% | 3.44 / 5.38 / 11.56 | 0.874 | 0.463 | 0.110 / 0.019 | 0.988 | 0.078 |
| a2f04_chamber_median | a2f04 | chamber | median | 10.9% | 3.44 / 5.10 / 8.15 | 0.864 | 0.485 | 0.122 / 0.085 | 0.996 | 0.082 |
| a2f04_box_shot | a2f04 | box | shot | 10.9% | 3.44 / 5.15 / 8.21 | 0.871 | 0.455 | 0.119 / 0.046 | 0.997 | 0.082 |
| **A: a2f04_chamber_shot** | a2f04 | chamber | shot | 10.9% | 3.43 / 5.15 / 8.21 | **0.870** | **0.458** | 0.119 / 0.047 | **0.997** | 0.082 |
| dmap_none_median | dmap | none | median | 15.9% | 5.30 / 7.24 / 11.38 | 0.701 | 1.800 | 0.173 / 0.117 | 0.985 | 0.085 |
| dmap_chamber_median | dmap | chamber | median | 10.7% | 4.94 / 6.41 / 10.02 | 0.352 | 1.800 | 0.170 / 0.164 | 0.986 | 0.085 |
| B: dmap_chamber_shot | dmap | chamber | shot | 10.7% | 4.90 / 6.37 / 9.59 | 0.315 | 1.823 | 0.161 / 0.139 | 0.986 | 0.088 |

The findings:

- **The region is the lever.** Dropped frames fall from 30.6% to 10.9%, and q95 correlation rises 0.778 -> 0.864 with
  the same median fill. The plain box gives the same 10.9% as the chamber, so the coil cut-out and line of sight add
  nothing on this flux. The box's inner edge (the O-point search erodes it by two cells) does the work.
- **The fill adds a little.** The median fill gives 0.864 and the shot fill 0.870. q95 vN halves from 0.085 to 0.047,
  near the masked file's 0.036. On the 10.9% of filled frames, the shot fill tracks the masked file closer: corr 0.771
  against 0.743, mean |diff| 1.14 against 1.39.
- **B is not a like-for-like scalar candidate.** The shape-map flux's plasma sits further inboard (boundary R0 median
  0.79 against 0.835 m) and is larger (a 0.56 against 0.54 m). eps ~0.70 then sits on the 0.65 cap for most frames,
  which lifts q95 by ~1.1 and flattens its variation (corr 0.31). Its boundaries correlate with A's at R0 0.93, a 0.89,
  kappa 0.63.

### DIII-D replay with labels (40 held-out scoring shots, 8,222 frames; `region-check`)

The MAST path is replayed on DIII-D:

- the physics flux (kappa 1.6, alpha 2) with its ellipse at the label (R0, a) on the midplane, standing in for MAST's
  Thomson placement;
- the gate is referenced to the label (R0, a), which is optimistic;
- volume pressure, eps cap 0.65.

| variant | filled | R2 q95 | R2 betaN | q95 corr with label-boundary q95 |
|---|---|---|---|---|
| label (EFIT) boundary, upper bound | - | 0.902 | 0.851 | 1 |
| scorer envelope, ungated (old MAST analogue) | 2.9% | -0.522 | 0.583 | 0.407 |
| FreeGS wall, ungated | 2.9% | -0.627 | 0.618 | 0.398 |
| no mask, gated, median fill (`22d99887` analogue) | 7.5% | -0.082 | 0.585 | 0.608 |
| no mask, gated, shot fill | 7.5% | -0.614 | 0.532 | 0.344 |
| FreeGS wall, gated, median fill | 5.1% | -0.210 | **0.635** | 0.601 |
| FreeGS wall, gated, shot fill | 5.1% | -0.635 | 0.613 | 0.402 |

The geometry region with the gate does not lose accuracy against the envelope analogue: q95 -0.21 against -0.52,
betaN 0.635 against 0.583. But q95 from this crude DIII-D flux is poor in every variant. Even on kept frames the
predictions average 3.59 against a true 5.15, and the kept frames' SSE exceeds the total SST.

On DIII-D the shot fill hurts q95. The dropped frames are early ramp frames (median position 0.13 into the shot).
Their in-shot donors are round, small ramp boundaries (kappa 1.04, a 0.72 of the label), and this multiplies the
filled frames' q95 SSE by 5 against the median fill. MAST does not show this pattern: its fill donors have median
kappa 1.68, and the fill tracks the masked proxy better there. So the fill choice is the one thing this check flags.
Only an official score can settle it.

### Files and recommendation

- New scalar files in `~/fusion/cache/mast_public_test/`: `scal_vol_eps065_{a2f04,dmap}_{region}_{fill}.npy` (the
  seven rows above), with boundary caches `lcfs_{psi stem}_{region}.npy`. The DIII-D boundaries are in
  `~/fusion/cache/d3d_heldout40_lcfs_wall.npy`.
- Submissions (MAST only, 5-shot validated):
  - `~/fusion/submission_mast_scal_A/` (psi_k1.6_a2.0_f0.4 + A);
  - `~/fusion/submission_mast_scal_B/` (psi_shape_dmap + B);
  - `~/fusion/submission_mast_scal_A_dmapflux/` (psi_shape_dmap + A).
- **Submit `A_dmapflux` first.** Against `dc37dc0a` (best compliant MAST flux, masked scalars) it changes only the
  q95 / betaN file. It therefore measures the cost of compliant scalars at the current best flux. A is the closest
  label-free match to the masked file (corr 0.870 against 0.778 for `22d99887`). If it lands below 0.320, the next
  single change is `a2f04_chamber_median` (the DIII-D-preferred fill).
- Logs: `~/fusion/logs/mast_scal_geom_20261012T083454Z.log` (scalar builds) and
  `mast_scal_geom_20261012T085052Z.log` (comparison, DIII-D check, builds), plus
  `mast_scal_d3d_fill_diag_*.log`.

### What is not established

- The official q95+betaN effect of any of these files. The masked file is a proxy for tracking, not truth.
- Whether MAST's eps really runs near the cap. B's level shift comes from its flux, and nothing label-free says which
  boundary size is right.
- Which fill is right on MAST. DIII-D and the MAST proxy disagree, and the DIII-D gate reference (label R0, a) is
  better than MAST's Thomson reference.

## Extraction-free MAST q95 / betaN and a Shafranov-shifted axis, 2026-10-13

Two levers, each gated by the DIII-D replay with labels (lesson from `1b4990da`: label-free MAST proxies misled us).
Rule 3 holds throughout. The replay reads DIII-D labels only. The MAST builds read coils, Ip, Thomson, machine geometry
and the DIII-D-fitted maps. Nothing was submitted or pushed.

### A: q95 / betaN from the placement and the shape map, no boundary extraction

[`placement_shape`](../../../fusioneq/shape_map.py) gives the q95 formula's (R0, a, kappa, delta) without extracting an
LCFS. (R0, a) is the plasma placement (Thomson extent on MAST, `plasma_geom.npy`), kappa is the ShapeMap's, and delta
is the mean of its tri_top and tri_bot. The pressure for betaN is averaged on ellipses centred at the map's Z_axis. The
formula and DIII-D calibration are unchanged (eps cap 0.65, volume pressure).

`geom-scal-check` ([`fusioneq/commands/shape.py`](../../../fusioneq/commands/shape.py)) replays every variant on
held-out DIII-D shots through [`ScalarReplay`](../../../fusioneq/physics.py). The extraction baselines are
`region-check`'s code, refactored into `ScalarReplay.extraction`, and reproduce its numbers exactly. Placement is the
label (R0, a) standing in for Thomson, as in every earlier replay. Because that is optimistic for A, which uses (R0, a)
directly, three harder placements are also run:

- seed 0 adds a Thomson-like error ([`perturb_placement`](../../../fusioneq/physics.py)): per shot, R0 + N(0, 2 cm) and
  a x (1 + N(0, 0.08)); per frame, a x (1 + N(0, frame sd)).
- frame sd 0.12 matches MAST's measured Thomson a jitter. That figure is label-free: the rms frame-to-frame change in
  log a is 0.12 on the public test, and the within-shot sd is 0.17.
- a x 0.9 / 1.1 is a systematic Thomson-against-LCFS bias.

The "160" set is the 160 non-scored held-out shots, less the ShapeMap's fit shots: 160 remain, 36,416 frames.

Mean R2 of q95 and betaN (the official `r2_qb`), with q95 / betaN R2 in brackets:

| placement | shots | label boundary (bound) | **A** | A, tri 0 | constant map shape | envelope (masked analogue, off. 0.320) | **no mask, median (22d99887, off. 0.208)** | wall, shot fill (1b4990da, off. 0.115) |
|---|---|---|---|---|---|---|---|---|
| label | 40 | 0.877 | **0.852** (0.820 / 0.883) | 0.729 | 0.814 | 0.030 | **0.251** (-0.082 / 0.585) | -0.011 |
| label | 160 | 0.855 | **0.783** (0.718 / 0.848) | 0.628 | 0.754 | -0.071 | **0.117** (-0.343 / 0.576) | -0.081 |
| seed 0, frame sd 0.04 | 40 | 0.877 | **0.664** (0.479 / 0.850) | 0.691 | 0.649 | 0.085 | **0.261** | -0.021 |
| seed 0, frame sd 0.04 | 160 | 0.855 | **0.559** (0.289 / 0.829) | 0.455 | 0.504 | -0.062 | **0.114** | -0.113 |
| seed 0, frame sd 0.12 (MAST jitter) | 40 | 0.877 | **0.513** (0.189 / 0.837) | 0.557 | 0.438 | 0.073 | **0.276** | -0.025 |
| seed 0, frame sd 0.12 (MAST jitter) | 160 | 0.855 | **0.309** (-0.195 / 0.814) | 0.275 | 0.221 | -0.071 | **0.134** (-0.319 / 0.587) | -0.131 |
| label, a x 0.9 | 40 | 0.877 | **0.726** | 0.437 | 0.704 | -0.016 | **0.250** | -0.017 |
| label, a x 1.1 | 40 | 0.877 | **0.639** | 0.812 | 0.602 | 0.077 | **0.256** | -0.022 |

**A passes.** It beats the mask-free extraction variant in all 8 replays. The margin falls from +0.60 to +0.18 as the
placement degrades. Almost all of it is betaN (0.81-0.92 against 0.58-0.60). q95 goes negative on the 160 set at
MAST-level jitter, though it stays above extraction's. The replay ranks the extraction variants the way the official
scores did: no mask with median fill above both the envelope analogue and the wall with shot fill. The map's
triangularity helps with a correct or small a, and hurts when a is 10% large.

**MAST label-free (disclosure, not selection).** `scal_vol_eps065_geom.npy` has q95 p5 / p50 / p95 1.26 / 5.73 / 9.71
and betaN 0.43 / 1.32 / 3.16. Its correlation with the masked file is 0.16 for q95 and 0.96 for betaN.

16.5% of frames have q95 < 2.5. On those frames the Thomson a is small (median 0.37 m against 0.53 m elsewhere), while
the masked file gives them 5.6. MAST's Thomson a does not correlate with our extracted a (r = -0.07). The q95 tail
is therefore Thomson-placement noise, heavier-tailed than the Gaussian jitter replayed above. This is the main risk
of A's q95 half. A physically implausible q95 < 2 could be screened by physics alone, but that would be a post-hoc
change and was not made.

### B: R_axis through a Shafranov-shifted axis

The DIII-D replay shows why R_axis consistency is -8. On frames that keep the fixed shape (5%), the scorer's derived
O-point misses by an sd of 42 cm. The derived axis also moves inward as the ladder's kappa_e rises: R0 + 6 cm at
kappa_e 1.0, R0 - 5 cm at 2.4, against a truth shift of +3 to +8 cm.

- [`ShiftModel`](../../../fusioneq/shape_map.py) fits Delta / a = c0 + c1 eps (beta_p + li / 2).
  - Delta is the scorer-derived R_axis of EFIT minus the labelled boundary centre.
  - beta_p comes from the Thomson volume pressure and Ip ([`shafranov_drive`](../../../fusioneq/scalars.py)).
  - li / 2 is held at 0.55, the DIII-D mean.
  - Predictions are clipped to [0, 0.5] a.
  - `shift-fit` fits it on the map's 800 fit shots (45,713 frames): c0 0.043, c1 0.195. On the 40 held-out shots
    Delta / a R2 is 0.288, and R_axis = R0 + a Delta / a has R2 0.520, against 0.295 for the constant mean shift.
- [`ShapeLadder.refine`](../../../fusioneq/mast_shape.py) now feeds the O-point back in R as well as Z. One step of up
  to 10 cm moves the ellipse centre toward R0 + a Delta / a. It is kept per frame only where the gate still passes
  and the miss shrinks. `shape-check --shift-axis` / `mast-shape --shift-axis`.

DIII-D replay (40 scoring shots, scored officially):

| replay | S | R2psi | D_LCFS | cons | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shape map (v2; rerun after the refactor reproduces it exactly) | 0.5126 | 0.7817 | 0.1732 | 0 | -8.01 | -9.22 | -3.48 | -0.56 | -0.54 | -12.06 | -0.42 |
| **+ Shafranov-shifted R feedback** | **0.5185** | **0.7917** | **0.1689** | 0 | **-5.94** | -9.21 | -3.80 | -0.45 | -0.46 | **-9.14** | -0.40 |

**B passes:** S +0.0059, from R2psi +0.010 and D_LCFS -0.004. R_axis improves (-8.0 to -5.9) and so does volume
(-12.1 to -9.1), but every consistency term stays below 0, so the consistency term itself does not move. kappa is a
little worse.

### Candidates and recommendation

- `~/fusion/cache/mast_public_test/scal_vol_eps065_geom.npy`, and `~/fusion/submission_mast_qb_geom/` (dc37dc0a's
  flux `psi_shape_dmap.npy` with these scalars, 5-shot validated). One change against `dc37dc0a`: the q95 / betaN file
  only. Against the compliant floor `22d99887` (0.208), the replay predicts a gain. Its size is uncertain because of
  the Thomson a tail above.
- `~/fusion/cache/mast_public_test/psi_shape_dmap_shift.npy`, and `~/fusion/submission_mast_shift/` (the shifted flux
  with `scal_vol_eps065_geom.npy`). One change against the A submission: the flux only. The S terms are separable,
  so R2psi, D_LCFS and consistency measure B alone.
  On MAST the R feedback was kept on 92.8% of frames and the Z feedback on 46.1%. The ellipse centre moved a median
  0.7 cm, and the derived R_axis median moved 0.894 -> 0.902 m. The label-free flags match v2: 0% LCFS failures,
  1.26% off-plasma axes, 3.77% kappa outside the band. The pick shares are unchanged.
- Submit A first, then B.
- Logs: `~/fusion/logs/mast_geom2_{A,B,D,E,subA,build_A}_*.log`. The replay arrays are
  `~/fusion/cache/geom_scal_check_*.npz` and `shape_check_{shift,rerun}.npz`. The model is `~/fusion/cache/shift_model.json`.

### What is not established

- **MAST's real Thomson-against-LCFS error.** The replay injects Gaussian error. MAST's q95 < 2.5 tail suggests a
  heavier tail, and its mean bias is unknown.
- **Whether the DIII-D Shafranov shift transfers.** On MAST the shift is measured from the Thomson density-extent
  centre, not the boundary centre, and that centre may already sit partly shifted. eps is about 0.6 against DIII-D's
  0.35, so the linear drive is extrapolated.
- **Official consistency.** The replay leaves it at 0 for both levers.
