# MAST (Challenge 2): what the demo shots showed, and what carried over officially

Challenge 2 ranks `G_ratio = S_MAST / S_DIII-D`, zero-shot: there is no MAST training set. MAST's flux comes from
physics ([`MastCache`](../fusioneq/physics.py)): the coil vacuum field plus an elliptical plasma current placed from the
Thomson density extent, calibrated on DIII-D; q95 / betaN follow from its boundary. Official rows are in
[`results/codabench.json`](../results/codabench.json).

## Official progression

| HF commit | change | MAST S | R2psi | q95+betaN | D_LCFS | cons | G |
|---|---|---|---|---|---|---|---|
| `df98320e` | physics flux, q95 eps cap 0.65 | 0.5883 | 0.873 | 0.204 | 0.245 | 0.009 | 0.5930 |
| `501b7986` | + volume-averaged betaN | 0.6017 | 0.873 | 0.294 | 0.245 | 0.009 | 0.6065 |
| `1497e518` | eps cap 0.7 instead | 0.6004 | 0.873 | 0.284 | 0.245 | 0.009 | 0.6051 |
| `d32bdf0c` | flux alpha 2.0, frac 0.4 (with v8 DIII-D) | **0.6045** | 0.877 | 0.320 | 0.261 | 0.000 | **0.6093** |
| `1a42cdea` | + per-frame psi offset fitted on the demo shots | **0.6367** | 0.936 | 0.320 | 0.261 | 0.000 | **0.6417** |

## A local check: the three labelled demo shots

The starter ships three MAST shots with full labels (`mast_shot_28348/50/51`, 115 frames). `mast-check` runs the same
pipeline on them (`--config mast_demo`) and scores with the official scorer. It ranks large effects correctly (volume
betaN, the alpha 2 flux) but not fine ones: it preferred eps 0.7 to 0.65 (officially the reverse) and predicted MAST
consistency 0.089 for the new flux (officially 0). Three shots set R2 against their own small variance, so only large
margins are trusted.

## Where the flux error is

Per term, physics flux against the labels on the demo shots: q95 tracks well (corr 0.96) but sits ~1.3 low; volume
betaN fixes most of betaN's offset (0.88 vs true 0.98, peak pressure 1.38); li, volume and R_axis carry signal at the
wrong level; kappa and triangularity cannot follow a fixed ellipse. A 36-setting sweep of current peaking (alpha),
Thomson threshold (frac) and kappa picked alpha 2.0 / frac 0.4 (demo S 0.537 -> 0.584), which transferred as R2psi
and q95+betaN gains.

Most of the remaining R2psi gap is a per-frame **level**, not shape:

| demo, best flux | R2psi |
|---|---|
| as built | 0.827 |
| + one global offset | 0.892 |
| + the right offset per frame | 0.957 |

That offset follows the solenoid current (corr -0.99) and the mean vacuum flux (+0.99); no single coil's scale explains
it, so it is not a units bug. It is the flux of vacuum-vessel currents, which the README says reconstructions include
and the inputs omit. [`SolenoidOffset`](../fusioneq/physics.py) fits offset = a + b * I_sol on the demo shots
(`mast-offset`, saved to `mast_offset.json`): leave-one-shot-out it scores R2psi 0.955 and demo S 0.654 against 0.827
and 0.584, matching the in-sample fit. **This is the one term fitted on MAST labels.** 17% of test frames lie slightly
outside the demo shots' solenoid-current range.

## Negatives

- Vacuum field at full strength instead of the DIII-D calibration's scale: demo R2psi 0.827 -> 0.805.
- Grad-Shafranov solve in MAST geometry (10-20 iterations, three current profiles): demo S 0.584 -> 0.564-0.566,
  D_LCFS and kappa worse, consistency back to 0, as on DIII-D. Without vessel currents or magnetics the free boundary
  relaxes to the wrong shape.
- Boundary distance across reasonable flux shapes varies by ~0.002 MAST S, below what three shots resolve.
- Richer flux-level models than a + b * I_sol (2026-10-07, leave-one-demo-shot-out): adding dI_sol/dt (vessel
  currents are induced by it), the plasma current, or all coil currents moves demo S 0.7070 -> 0.7070-0.7072 and
  R2psi 0.9414 -> 0.9418, inside the noise; 64-65% of test frames sit outside the demo range of dI_sol/dt and Ip, so
  the fit would extrapolate. The solenoid offset already takes what three shots can teach about the level.
- Machine-agnostic grid model, stopped at gate 0 (2026-10-10). The plan: train a grid shape model on DIII-D from inputs
  that mean the same on both machines (vacuum-flux shape, the physics flux with the plasma placed from Thomson, and
  plasma-normalised coordinates), then check it on the demo shots. Gate 0 asked whether DIII-D's plasma can be placed
  from Thomson the MAST way. It cannot: DIII-D's core chords are vertical and cover only the upper half (MAST's are
  radial), so a Thomson estimate (top from the core profile, outer edge from the tangential chords, Z0 = 0, kappa
  1.6) is found in 99.9% of frames but barely tracks EFIT (R0 corr -0.16, a corr 0.19, a biased 0.44 m vs 0.59 m).
  Placement does not matter to the physics flux on DIII-D either (label-placed S 0.504, Thomson-placed 0.505, with
  the E-coil term). With no machine-independent plasma descriptor, an agnostic model can only learn a typical DIII-D
  plasma - the imprinting that sank the learned flux correction.
- Plasma-size remap a' = med + s (a - med) + d on the Thomson minor radius (2026-10-10). On the demo shots our volume
  tracks the truth (corr 0.88) but runs low (7.3 vs 8.1) and twice as wide, while R_axis and li (corr 0.72, 0.74) are
  the only scalars with positive R2; kappa, triangularity and Z_axis are uncorrelated. Remapping a (16 settings,
  solenoid offset refitted) barely moves volume (R2 -4.6 -> -4.3 at best): the scorer's boundary, and so its volume,
  is set by the vacuum field rather than the plasma ellipse. The best setting (d = +0.03 m) lifts li and R_axis, demo S
  0.7298 -> 0.7336, but leave-one-shot-out is mixed (-0.003 / +0.006 / +0.004). Not submitted. The level-type fixes
  are harvested; MAST's remaining shape and volume error needs a learned MAST model, and there is no MAST training data.
- Rule 3 (checked 2026-10-10): the demo shots are for inspection and visualization only, so the solenoid offset and
  the q95 / betaN lines cannot be in Final submissions. Label-free baseline (`62f9e89f`, MAST as `d32bdf0c`): MAST
  0.604496, G 0.608286. Half-solenoid hypothesis (`f2b36e3e`; the Data page notes `magnetics_sol_current` reads 2x
  FAIR-MAST level-2): MAST 0.54068 - R2psi 0.877 -> 0.750 (level worse) but D_LCFS 0.2606 -> 0.2198 and consistency 0 ->
  0.0117 (shape better). The factor does not explain the flux-level error.
- The shape part alone (`4c428222`: each frame's half-solenoid correction minus its grid mean, so the level is kept):
  MAST 0.627752, G 0.631688 - R2psi 0.877 -> 0.908, D_LCFS 0.2197, consistency 0.0117. The best label-free MAST so far,
  recovering ~58% of the banned offset's gain; chosen by official feedback (disclosed in the methods report).
- Coil geometry audit (2026-10-11, geometry only): the shipped MAST coil description matches published machine
  descriptions (FreeGS `MAST`) to within a few cm for P2-P6 and the solenoid (656 elements, R 0.12-0.16, Z +-1.47 vs
  R 0.15, Z +-1.40). So the solenoid results point at its current, not its geometry: halving its flux shape helps
  (consistent with the Data page's 2x caveat), halving its level hurts - a second I_sol-proportional level term
  (vessel currents are the usual suspect) is what the banned demo offset captured, and nothing label-free calibrates
  it. Label-free MAST is near its ceiling at 0.628 short of a model trained on synthetic MAST equilibria.
- Free-boundary Grad-Shafranov, centroid pinned or free (2026-10-11, checked on DIII-D where labels are allowed, so no
  rule-3 issue): the Picard solve ([`GradShafranov`](../fusioneq/equilibrium.py), `gs-check --free`) cannot make the
  shape scalars track EFIT even on DIII-D. Pinned at the ellipse centroid: S 0.5035, consistency 0 with every scalar's
  R2 far below zero (R_axis -9.2, kappa -11.4); letting the centroid move makes it worse (S 0.392, R_axis -37,
  kappa -19), since nothing holds the plasma vertically without magnetics or vessel currents. Higher-ranked teams'
  MAST consistency (0.24-0.89) therefore does not come from a self-consistent solve with our inputs alone; this
  route is closed.

## Coil geometry vs FreeGS `MAST()`, quantified (2026-10-11, geometry and test inputs only)

Lever 4 of the [SOTA sweep](../reports/2026-10-11-fusion-equilibrium-sota.md) asks if the shipped MAST geometry is
short (turns, height, passive conductors), since an error there would show up as a flux level proportional to I_sol.
The reference is FreeGS [`MAST()` / `MAST_sym()`](https://github.com/freegs-plasma/freegs/blob/5b41fe5565ff8077bf8139051c80c69416fc9b86/freegs/machine.py#L1408-L1456)
(master `5b41fe5`, 2026-10-05). Unit-current fluxes on the 65x65 MAST grid are compared with
[`BasisDiff`](../fusioneq/mast_geometry.py), which splits the difference into a grid-mean level and the rest (shape).
Run it with `mast-geometry`. No labels, demo shots or scorer masks are read. The I_sol figures come from the inputs of
41 test shots (2,245 frames, every 30th file): I_sol p5/p50/p95 = -44 / -24 / +5 kA. They use |calibration scale|
0.8724.

**What each side describes.** The shipped table has 812 conductor elements from FAIR-MAST `pf_active`: P2 = 12 inner
+ 8 outer, P3 8, P4 23, P5 23, P6 4, and the solenoid as 656 turns (4 layers at R = 0.119 / 0.132 / 0.145 / 0.157,
164 turns each, Z -1.471..+1.459). FreeGS is much coarser. It uses one single-turn filament per P-coil. Its solenoid
is 100 filaments at one radius (R 0.15, Z +-1.40, or +-1.45 in `MAST_sym`). It has no wall, vessel or passive
conductors. The 100 turns are a placeholder: FreeGS gives no turn counts anywhere in `MAST()`, while its `MASTU()`
gives the MAST-U solenoid 324 turns.

| column | mean flux, shipped / FreeGS | shape diff / shipped rms | note |
|---|---|---|---|
| P3 U/L | 1.004 / 1.005 | 0.011 / 0.016 | matches |
| P4 U/L | 0.990 / 0.991 | 0.034 / 0.031 | |
| P5 U/L | 0.995 / 0.996 | 0.041 / 0.050 | |
| P2 U/L | 1.039 / 1.037 | 0.10 / 0.11 | inner + outer stack vs one filament at (0.49, +-1.76), at the grid's edge |
| P6 U/L | 0.965 / 0.965 | 0.10 / 0.10 | shipped centroid R 1.44 vs FreeGS 1.50 |
| solenoid vs `MAST()` (N 100) | **5.62** | 0.82 | the gap is the turn count |
| solenoid vs `MAST_sym()` (N 100) | 5.65 | 0.82 | |
| solenoid vs FreeGS R/Z with the shipped 656 turns | 0.857 | 0.19 | radius 0.15 vs shipped rms 0.139 |

Solenoid bounds within the shipped description, holding the shipped turn heights fixed:

| radial placement of all 656 turns | mean flux / shipped | shape change / shipped shape std |
|---|---|---|
| element centres, no 3x3 sub-filaments | 1.000 | 0.002 |
| innermost layer (0.119) | 0.740 | 0.29 |
| FreeGS radius (0.150) | 1.161 | 0.18 |
| outermost layer (0.157) | 1.274 | 0.30 |

**Converting to a per-frame level.** On the grid, our solenoid gives 1.60e-3 Wb/rad per kA averaged over the grid.
Its shape varies by 0.64e-3 per kA (std), so it is not a pure level. Over the test frames, the solenoid's calibrated
grid-mean level has a std of **0.022 Wb/rad** (median -0.033). Each geometry option changes that level by this much,
as a per-frame std:

- FreeGS's radius 0.15 with the shipped turns: +16%, about **0.0037 Wb/rad** (median +0.0056).
- Any radial placement within the shipped layer span: -26% to +27%, at most **0.006 Wb/rad**.
- FreeGS's 100 turns: -82%, 0.018 Wb/rad. This is not a candidate, because those turns are a placeholder, not a
  measurement.
- P-coils: level ratios are 0.965-1.039 per kA-turn. Any level they add follows the P-coil currents, not I_sol, so
  they cannot produce the solenoid-correlated term.

**Verdict: closed, not submit-worthy.** FreeGS does not show the shipped geometry to be short. The shipped description
is the finer of the two: it is turn by turn from the operator's own `pf_active`, against single filaments and a
placeholder turn count in FreeGS. The two agree within 1-4% on P3-P5, and P2 and P6 differ only by representation.
FreeGS models no vessel or passive conductors, so it gives no geometry for the vessel-current hypothesis. Moving the
solenoid's radius is the only change left. It is bounded at +-27% of our solenoid level (frame std at most 0.006
Wb/rad), and FreeGS's own radius sits inside the shipped layer span, so it gives no grounds for picking a direction.
Every such move also changes the solenoid's shape (0.18-0.30 of its shape std), so it is not a clean level term.
Picking a radius or turn factor from Development scores would be a one-constant level sweep: a probe, not a geometry
fix. The remaining I_sol-proportional level stays unexplained by geometry. The Data page's 2x solenoid-current caveat
(a 50% level factor, already tested: it worsens the level) is larger than anything geometry can supply.

## Induced vessel and passive currents, label-free (2026-10-14, inputs and published geometry only)

The audit above clears the coil geometry, which leaves induced currents in the vessel and passive structures as the
suspect for the solenoid-tracking flux level. This section asks whether a circuit model built only from published
constants can produce that level. No demo labels, scorer masks or `mast_offset.json` were read. The target numbers
below come from the tables above.

**What is published.**
- MAST-U EFIT++ models the conducting structures as 150 passive elements: vessel, centre column, supports, gas baffles
  and coil cases. Graphite tiles and the toroidally broken cryopump are left out, and the paper notes that in a
  spherical tokamak these eddy currents strongly affect shape and position
  ([FreeGSNKE / Fiesta vs EFIT++ on MAST-U, arXiv:2407.12432](https://arxiv.org/abs/2407.12432)).
- MAST's kinetic EFIT reconstructions use **18 vessel segments**, among them the P6 cases and the divertor plates near
  P2. They are described in Berkery et al.,
  [UKAEA-CCFE-PR(21)79](https://scientific-publications.ukaea.uk/wp-content/uploads/UKAEA-CCFE-PR2179.PDF) (PPCF
  2021). A 3-D VALEN model of the vessel with its ports, the centre column and the coil cases was calibrated on vacuum
  coil test shots. In reconstruction, each segment's current is estimated as **a nearby loop voltage divided by an
  effective resistance**, for example 1.875e-5 Ohm for segment 7U. The paper calls the induced current "significant"
  against a ~750 kA flat-top Ip (shot 23822). It says the current is driven "especially [by] the +22 to -17 kA swing
  in the P1 Ohmic solenoid". On MAST-U vacuum shots it reaches about 90 kA, in the outer vessel early and in the
  centre-stack casing during the Ohmic swing.
- So the reference psi does contain vessel-current flux that our coil-only model lacks, as the starter README also
  says. Because each segment current is loop voltage over resistance, it is resistive: it follows dpsi/dt, not the
  solenoid current itself.
- We found no published L/R time constant for the MAST vessel. The only machine-readable geometry with resistivities
  is FreeGSNKE's `MAST-U_like_passive_coils.pickle`
  ([FusionComputingLab/freegsnke@6d43a09](https://github.com/FusionComputingLab/freegsnke/tree/6d43a09/machine_configs/MAST-U),
  source "UKAEA"). It holds 138 polygons with resistivities of 7.1e-7 (vessel) and 7.4e-7 Ohm m (everything else,
  i.e. stainless steel). The vessel is a 20 mm wall at R 2.00-2.02 m with 22 mm lids at Z +-2.21 m, a 12 mm centre
  tube at R 0.24-0.30 m, and cases on P4, P5 and P6. MAST-U kept MAST's vessel, but its divertor coil cases, gas baffles
  and plates are MAST-U-specific, so we use two subsets: the 22 vessel polygons alone, and those plus the centre column
  and the P4/P5/P6 cases (58 elements, "MAST-common").

**Model.** [`PassiveSet`](../fusioneq/vessel.py) treats each element as an axisymmetric ring and integrates
L dI/dt + R I = -2 pi dpsi_ext/dt by implicit Euler at 0.2 ms. Mutual inductances use the same filament Green's function as
[`greens.py`](../fusioneq/greens.py), self-inductances Grover's ring formula, and resistance rho 2 pi R / area from the
polygon area. psi_ext is driven by the shipped solenoid (656 turns) and the P-coils from FAIR-MAST `pf_active`, plus
Ip as a filament at R 0.85 m. Integration starts from rest 220-300 ms before the first EFIT time, more than 6 of the
longest time constants. The result is each frame's grid-mean passive flux on the 65x65 grid. The probe scripts are
one-offs and are not committed. The Data page's 2x solenoid caveat would halve the solenoid-driven part but leave its
correlations unchanged.

**What the term would have to do.** On the demo shots the missing per-frame level correlates -0.99 with I_sol. A
per-frame offset over a global one lifts R2psi by 0.065, which is about 0.0125 Wb/rad of per-frame std given that our
predicted psi has a 0.049 Wb/rad pixel std on test. That conversion is rough: it uses our prediction's spread in place
of the truth's. For scale, our solenoid's level std is 0.022-0.026 Wb/rad.

**Result.** 41 public-test shots, every 30th file, 2,245 frames. I_sol p5/p50/p95 = -44 / -22 / +5 kA and dI_sol/dt =
-375 / -137 / +74 kA/s.

| passive set | longest L/R | median total current | level std (within shot) | corr with I_sol | corr with dI_sol/dt | corr with Ip |
|---|---|---|---|---|---|---|
| vessel (22) | 32 ms | 32 kA | 0.0073 (0.0041) | **-0.04** | -0.37 | +0.71 |
| vessel + centre tube + P4-6 cases (58) | 33 ms | 110 kA | 0.0156 (0.0087) | **-0.13** | -0.27 | +0.68 |

Scaling every resistivity across four decades (a stand-in cylinder, 2-2,250 ms) shows when the level would follow
I_sol. It needs a time constant longer than the shot, at least ~0.75 s, i.e. at least 30x below steel's resistivity.
Even then it reaches only corr -0.74 to -0.90 and std 0.012-0.015, by carrying about 260 kA, which is a third to half
of Ip. Steel at its published resistivity gives about 30 ms. That is far shorter than the ~300 ms solenoid swing, so the
induced level follows dI/dt and the plasma current, not I_sol.

**DIII-D analogue (labels allowed).** 281 training shots, every 25th, 62,280 frames. Each frame's EFIT grid-mean psi
was fitted from the vacuum level, ECOILA and Ip, with 5-fold cross-validation grouped by shot (CV R2 0.7249). Adding
dECOILA/dt, dIp/dt or both gives 0.7240-0.7245, and the leftover level correlates 0.03 with dECOILA/dt. DIII-D's EFIT
also fits vessel currents, yet at its 20 ms frame spacing no dI/dt level shows up.

**Verdict: do not build.**
- The published-constant passive model induces 32-110 kA, the same order as the published ~90 kA, but its flux level is
  uncorrelated with I_sol (-0.04 to -0.13 against the needed -0.99). The reconstructions' own vessel currents are loop
  voltage over resistance, so they follow dI/dt in the same way.
- The only sizeable part of it follows Ip and dI/dt. Fitting Ip and dI_sol/dt into the demo offset already moved
  nothing (2026-10-07, above).
- The configuration that would track I_sol needs a time constant 30x too long and a vessel current near the plasma
  current. Nothing published supports that.
- Adding the term as-is would be an unvalidated ±0.007-0.016 Wb/rad change, and only Development scores could judge
  its sign, which is a rule-3 probe.

What would change this: a published MAST vessel-mode time constant or VALEN-fitted group resistances longer than about
0.5 s (none found), or the geometry and resistances of MAST's 18 EFIT segments (published for only one segment).
Without one of these, the solenoid-tracking level stays unexplained, and the label-free MAST flux stays at R2psi 0.908
(`4c428222`).

## MAST li and kappa: why they are negative, and a core-flux li lever (2026-10-14, DIII-D labels + MAST inputs only)

Official `688a773d` (best compliant MAST, S 0.619443) scores li R2 -1.362 and kappa R2 -0.265. Each consistency term
is clipped at 0 before averaging, so these two contribute nothing now. They only add to S once they turn positive:
each +0.1 R2 on one scalar is worth +0.1 / 7 x 0.20 = +0.0029 S. This section diagnoses both from MAST inputs and
published ranges, then tests one lever, a core-flux remap toward a target li, on the labelled DIII-D replay. No MAST
labels, demo shots, scorer masks or probe scores were read.

**Published MAST ranges** (primary sources; spreads read from figures, since no paper states a mean or sd):
- li(2), named explicitly: the time-weighted operating-space density peaks at li(2) 0.8-0.95 and kappa 1.85-2.0.
  Support is li(2) 0.45-1.75 and kappa 1.55-2.6, and kappa falls as li rises
  ([Stork et al., IAEA FEC 2010 ICC/P5-06, Fig. 2](https://www-pub.iaea.org/MTCD/meetings/PDFplus/2010/cn180/cn180_papers/icc_p5-06.pdf)).
  Table 1 gives maximum kappa 2.5 and A ~1.3.
- Kinetic + MSE EFIT, 275 times in 10 shots: li 0.58-0.95, with an example of 0.848, "typically ... down to 0.6"
  ([Berkery et al., PPCF 63 (2021) 055014, Fig. 11](https://scientific-publications.ukaea.uk/wp-content/uploads/BERKERY_2021_PLASMA_PHYS-_CONTROL-_FUSION_63_055014.PDF)).
- The DECAF disruption database (ramp-up excluded) spans li 0.45-2.4 and kappa 1.3-2.75. It notes MAST "did not often
  access high elongations (kappa > 2.2)"
  ([Berkery et al., UKAEA-CCFE-CP(23)15, Fig. 1](https://scientific-publications.ukaea.uk/wp-content/uploads/UKAEA-CCFE-CP2315.PDF)).
- The early operating space has its densest ridge at li 0.8-1.2
  ([Hole et al., EPS 2002 O-2.02, Fig. 3](https://lac913.epfl.ch/epsppd3/2002/pdf/O2_02.pdf)).

**Our MAST values.** These are the scorer's derivation of the served flux `psi_shape_dmap_shift.npy`, with no mask, on
every 10th public-test frame (6,908 frames). li derives on 98.8% of them.

| | li p5 / p50 / p95 | li robust sd | kappa p5 / p50 / p95 |
|---|---|---|---|
| all frames | 0.787 / 1.365 / 2.168 | 0.293 | 1.667 / 1.790 / 1.941 |
| flat-top, Ip >= 0.8 shot max (5,681) | 0.983 / 1.379 / 2.190 | 0.282 | 1.683 / 1.788 / 1.866 |
| published typical | ~0.85-0.95 (density peak) | ~0.15 (figure) | ~1.85-2.0, core 1.6-2.2 |

**Why li is negative: level and spread are both wrong, and the spread is mostly noise.**
- The level sits about +0.45 above the published li(2) peak.
- The spread is about twice the published one.
- On flat-top frames our li correlates -0.66 with the Thomson minor radius. The correlations with Ip (-0.01) and shot
  phase (-0.21) are weak.
- The plasma current sits in an ellipse sized by Thomson, and the LCFS is set by the coils. A small Thomson a leaves a
  narrow current channel inside a fixed LCFS, so the edge field drops and li rises. The Thomson a jumps ~12%
  frame-to-frame (label-free; see the Thomson jitter in
  [`mast_shape.md`](experiments/fusioneq/mast_shape.md)), so most of our li variation is placement noise, not current
  profile.
- On the DIII-D replay (40 scoring shots), where placement comes from the labels, the same flux gives li median 1.19
  against a true 1.06 (+0.14), correlation 0.44 and R2 -0.40. MAST's bias is about three times larger.

**Why kappa is negative: a near-constant prediction, set a little low.**
- Our flat-top kappa spans 0.18 (p5-p95) against a published core range of about 0.6. The median, 1.79, sits about 0.1
  below the published ~1.9.
- A constant 0.1 off the truth's mean, with sd ~0.2, scores R2 about -(0.1/0.2)^2 = -0.25. That matches the official
  -0.265 without any frame-to-frame tracking.
- The tails are small on flat-top (p99 2.40). Over all frames, 3.6% sit above 2.5 and 1.2% below 1.3, mostly in ramps.
- A positive kappa R2 needs per-frame tracking. The ShapeMap predicts DIII-D kappa at R2 0.43, but realising it in the
  physics flux gives -3.8 on the replay ([`mast_shape.md`](experiments/fusioneq/mast_shape.md)). Moving kappa also
  moves the LCFS (D_LCFS). No label-free kappa lever was found within this spike. Raising the level alone could at best
  bring the clipped term to 0, which gains nothing.

**Lever tested: core-flux remap to a target li**
([`CoreShaper`](../fusioneq/li_shape.py), commands `li-check` / `mast-li` in
[`fusioneq/commands/li.py`](../fusioneq/commands/li.py)).
- Inside the flux's own LCFS (no envelope), with x the normalised flux (0 on the LCFS, 1 at the axis):
  phi -> phi + c (phi_axis - phi_b) x^2.
- The remap keeps the LCFS, the nesting of the flux surfaces, the boundary slope (so the Ip read from the boundary is
  unchanged) and the axis cell. Only the core, R2psi and li move.
- The scorer's li(2) is tabulated over c in [-0.45, 1.5] per frame, and c is interpolated to the target
  li = level + k (li - our median li).
- Tests: [`tests/test_li_shape.py`](../tests/test_li_shape.py). The tabulated li matches the scorer's to 1e-9, the
  remap keeps the extracted LCFS, and the target is hit.

**DIII-D replay** (the shifted MAST procedure on the 40 held-out scoring shots, 8,222 frames, scored officially;
baseline S 0.5185, R2psi 0.7917, li R2 -0.404). D_LCFS is unchanged at 0.1689 in every row. The true li median is
1.058; ours is 1.193.

| level | k | median c | S | dS | R2psi | li R2 (scored) |
|---|---|---|---|---|---|---|
| own (1.19) | 0.5 | 0.00 | 0.5188 | +0.0003 | 0.7922 | -0.002 |
| own | 0.25 | 0.00 | 0.5189 | +0.0004 | 0.7923 | -0.010 |
| own | 0 | 0.00 | 0.5189 | +0.0004 | 0.7924 | -0.157 |
| 1.15 | 0.25 | -0.02 | **0.5207** | **+0.0022** | 0.7921 | **+0.068** |
| 1.05 | 0.5 | -0.09 | 0.5199 | +0.0014 | 0.7912 | +0.057 |
| 1.05 | 0.25 | -0.09 | 0.5196 | +0.0011 | 0.7913 | +0.044 |
| 1.05 | 0 | -0.09 | 0.5182 | -0.0003 | 0.7911 | -0.107 |
| 0.95 | 0.25 | -0.16 | 0.5176 | -0.0009 | 0.7900 | -0.265 |
| 0.85 | 0.25 | -0.24 | 0.5167 | -0.0018 | 0.7884 | -0.861 |
| 0.85 | 0 | -0.23 | 0.5165 | -0.0020 | 0.7880 | -1.014 |

What the replay shows:
- li turns positive only in a narrow window of level, from the true median to about +0.1 above it. The best rows lift
  the scored li R2 to +0.06-0.07.
- Undershooting by 0.1 already costs S. Shrinking the spread (k 0.25-0.5) helps only once the level is right; a flat
  target (k 0) never scores above 0.
- Flattening the core costs R2psi, even toward the true level: -0.0005 at 1.05, -0.0037 at 0.85.
- The best gain, +0.0022 at level 1.15, needs the level placed from DIII-D labels. It still lowers R2psi (-0.0004).

**MAST would need a far larger move.** `mast-li --level 0.9 --k 0` asks for a median c of -0.28 (p5 -0.45, p95
+0.72). 17.1% of frames sit at the grid floor (-0.45), where the remap cannot flatten further. Reaching published
levels moves the core more than the DIII-D 0.85 rows did, and those cost R2psi -0.004. The published level is also
uncertain by more than the replay's window: Stork's peak is 0.8-0.95, Hole's ridge is 0.8-1.2, and the test set
includes ramps. And the scorer's grid li(2) on MAST EFIT is not shown to equal EFIT's own li. The expected MAST dS is
about zero or negative.

**Verdict: no candidate.** The gate was +0.002 S without hurting R2psi or D_LCFS. It is not met: the best DIII-D gain
(+0.0022) needs a label-placed level and still lowers R2psi. The label-free MAST level needs a larger move with an
uncertain target. Nothing was built under `~/fusion/submission_mast_li/`. The diagnostic flux
`~/fusion/cache/mast_public_test/psi_shape_dmap_shift_li_l090_k0.npy` (with its `_record.npz`, and the cached
`psi_shape_dmap_shift_li_{bumps,curves}.npy`) is not a submission candidate.

What would change this:
- a label-free current-profile signal that tracks li, such as MSE-free magnetics (beta_p + li/2 from the diamagnetic
  loop and Shafranov integrals, not among the shipped inputs) or a Thomson Te-peakedness predictor. Any such signal
  must reach li R2 > 0.2 on the DIII-D replay.
- or a placement with less jitter than the Thomson a, which would cut the noise in our own li.

**Not established:**
- MAST's true li level under the scorer's grid derivation.
- Whether our MAST li correlates with the truth at all: the DIII-D correlation is 0.44 with label placement.
- How R2psi responds on MAST to a core flattening of this size.

Logs: `~/fusion/logs/li_replay_20261014T112759Z.log` (replay rebuilt; reproduces S 0.5185 exactly),
`li_check_20261014T113556Z.log`, `li_mast_20261014T113246Z.log`, `li_mast_diag_20261014T112605Z.log`. Arrays: `~/fusion/cache/shape_check_shift_psi.npy` (replay
flux), `li_check.npz`, `~/fusion/li_spike/mast_diag.npz`.
