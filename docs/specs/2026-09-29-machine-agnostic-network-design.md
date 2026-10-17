# Machine-agnostic network for MAST transfer — design

**Status:** proposed, 2026-09-29. **Goal:** raise MAST S (and so G_ratio, Award 2) from 0.575 toward the 0.9+ the
Award 2 leaders reach, without MAST equilibrium data. DIII-D submissions keep using the PCA-MLP ensembles.

## Why

The leaderboard (development phase, 2026-09-29) shows three teams at MAST S 0.94-1.00 and many near 0.73; we are at
0.575 with a physics-only flux ([`physics.py`](../../fusioneq/physics.py)). The Grad-Shafranov solve
([`equilibrium.py`](../../fusioneq/equilibrium.py)) gained only +0.002 on DIII-D because the remaining error is in the
external field, not the current placement. The teams at 0.9+ almost certainly transfer a *learned* model. Ours cannot:
[`NeuralPredictor`](../../fusioneq/model.py) reads DIII-D's 20 coil columns and 40 Thomson stations on DIII-D's chord
layout, none of which exist on MAST.

Our MAST score today:

| term | weight | MAST value | lost |
|---|---|---|---|
| R2psi | 0.55 | 0.873 | 0.070 |
| q95, betaN | 0.15 | 0.116 | 0.133 |
| 1 - D_LCFS | 0.10 | 0.755 | 0.024 |
| Consistency | 0.20 | 0.009 | 0.198 |

Consistency and q95/betaN are 80% of the loss. Both follow from getting the plasma *shape* and *current profile* right.

## The two machines

| | DIII-D | MAST |
|---|---|---|
| grid (65 x 65) | R 0.84-2.54, Z +-1.60 m | R 0.06-2.00, Z +-2.00 m |
| scorer envelope centre / half-width | R 1.68 / 0.70 m | R 1.01 / 0.86 m |
| plasma R0 / a (aspect ratio) | ~1.68 / 0.6 (2.8) | ~0.82 / 0.56 (1.5) |
| Ip, median | 0.99 MA | 0.63 MA |
| coils | 18 F-coils + E-coil + B-coil (20 columns) | 10 P-coils + solenoid + EFPS + TF (13 columns) |
| core Thomson | vertical chord at R ~ 1.94 m, 44 points | midplane, radial, R 0.3-1.45 m |

Anything the network reads must exist on both, in comparable units. MAST is a spherical tokamak: training only on
DIII-D means extrapolating in aspect ratio, which is the central risk.

## Design

### 1. Predict the current, not the flux

The network outputs a toroidal current density **J(R, Z) >= 0 on the machine's own 65 x 65 grid, normalised to the
measured Ip**, and the flux follows from exact physics:

    psi = s * (psi_vac + G J) + c

- `psi_vac` is each machine's coil flux ([`coil_basis`](../../fusioneq/greens.py)); `G` is its grid Green's matrix
  ([`grid_greens`](../../fusioneq/greens.py)). Both already exist for both machines.
- `s, c` are the global scale and offset ([`Calibration`](../../fusioneq/physics.py)), fitted on DIII-D.
- A softmax over envelope cells times Ip makes the total current exact on any machine.

This keeps what transfers today (MAST R2psi 0.873 comes from `psi_vac + G J_ellipse`) and learns only the part that
was wrong: where the current sits and how it is shaped. A current blob is far more machine-agnostic than a flux map.

### 2. A common, machine-normalised frame

Resample each frame's input fields onto a 128 x 128 grid in **envelope-normalised coordinates**
`x = (R - R_c) / h`, `y = Z / h` (R_c, h = the scorer envelope's centre and half-width). The network runs there, and its
output is resampled back to the machine grid before the `G J` step. Input channels, all dimensionless:

| channel | definition | why |
|---|---|---|
| vacuum flux | `psi_vac / (mu0 Ip R_c)` | the external field shape; units cancel between machines |
| toroidicity | `R / R_c` | carries aspect ratio explicitly (MAST's inner edge sits at R/R_c ~ 0.15, DIII-D's at ~0.58) |
| envelope | scorer mask | where plasma can be |
| vertical position | `y` | up-down asymmetry |
| scalars (broadcast) | `mu0 Ip / (a B0)`, Thomson peak ne, Te, pressure / B0^2 | normalised current, density, beta; B0 from the TF current ([`scalars.py`](../../fusioneq/scalars.py)) |

Thomson enters only as **machine-independent scalars** (peak ne, peak Te, peak pressure, and the density extent along
the chord, normalised by h). The chord geometries are incompatible, so no profile shapes.

### 3. Architecture and loss

- A small U-Net (4 levels, ~2-5 M parameters) on the 128 x 128 frame, input = channels above for the frame and its +-2
  neighbours (temporal context helped on DIII-D). Output = J logits.
- Loss on the machine-grid psi: the mix loss analogue (pixel MSE plus MSE inside the envelope), plus a small
  q95/betaN head loss. Gradients pass through the fixed `G` (a 4225 x 4225 matmul per frame, cheap on the A100).
- Train on DIII-D only, same 200-shot hold-out and 40 scoring shots, so DIII-D scores stay comparable.

### 4. Transfer aids

- **Size augmentation.** Rescaling a whole machine by s with currents fixed scales flux by s. Randomly rescaling DIII-D
  frames (grid, envelope, psi, psi_vac by s in [0.5, 1.2]) teaches size invariance for free.
- **Aspect-ratio augmentation (stretch goal).** Shifting R changes toroidicity, and psi_vac for shifted coils can be
  recomputed exactly from the coil table, but EFIT targets cannot. Would need synthetic equilibria from the GS solver.
- **q95 from the predicted flux.** q95 = (1/2 pi) closed integral of B_phi / (R B_p) dl on the 95% surface. B_p comes
  from the predicted psi and B_phi from the TF current, so q95 follows from psi with no learned transfer. betaN stays a
  learned head on the normalised Thomson scalars.

## Validation

No MAST labels, so three layers:

1. **DIII-D held-out S** as a floor. The agnostic net only needs about 0.97 here; its job is MAST.
2. **MAST sanity checks on inputs only:** the official LCFS extractor closes on most frames, R_geo and elongation in
   MAST's range (R_geo ~ 0.8-0.9 m, kappa ~ 1.8-2.2), the current centroid inside the Thomson density extent.
   [`boundary_shapes`](../../fusioneq/score.py) and the `mast-physics` checks already do this.
3. **Codabench** as the MAST score: one slot per serious candidate, alongside the DIII-D ensemble (the pointer
   submits both files).

## Milestones

| | work | exit test |
|---|---|---|
| M0 | normalised-frame resampling, channels and a cached `G` per machine | round-trip resampling error < 1e-3 of the psi range. **Done**: [`MachineFrame`](../../fusioneq/agnostic.py), `frame-check`; 128 x 128 bicubic gives p50 2e-4 (DIII-D vacuum), 4e-4 (DIII-D EFIT psi), 8e-4 (MAST vacuum); only MAST's p99 (2.4e-3, beside the solenoid) misses, and psi never passes through the frame. The normalised vacuum channel is comparable across machines (median -0.09 DIII-D, -0.12 MAST). |
| M1 | physics baseline in the new pipeline (J = ellipse) | reproduces MAST psi_phys and DIII-D physics-check scores. **Done**: `pipeline-check`; DIII-D S 0.5014 vs 0.5013 (with E-coil) and 0.1326 vs 0.1325 (without); MAST R2 1.000000 against `psi_phys.npy`, max error 5e-4 (p50) / 7e-4 (p99) of range, the float16 storage floor. |
| M2 | U-Net J-predictor trained on DIII-D | DIII-D held-out S >= 0.97; MAST sanity checks pass. **Failed** (`agnostic_m2_i00`, 4 epochs, 43 min): S 0.673 with borrowed q95/betaN, psi-only 0.523 vs the ellipse's 0.501 (R2psi 0.799, 1-D_LCFS 0.838, consistency 0, li -10); loss flat at ~3.0. MAST: 100% closed, R_geo 0.80 m, kappa 1.74. The current is well placed (DIII-D centroid within 1 cm of the axis in Z; MAST 99% inside the Thomson extent), but 79% of the psi MSE is a per-frame offset no current can fix. Fitting 6 vacuum fields per frame from labels (an upper bound) lifts R2psi to 0.968 yet leaves consistency at 0, so the shape error is in the current's *shape*, not the external field. |
| M2b | offset-free shape loss inside the envelope + a learned offset scalar; 8 epochs | same exit test. **Failed** (`agnostic_m2b_i00`, 8 epochs, ~50 min): S 0.774 with borrowed q95/betaN, psi-only 0.625 (M2 0.523, ellipse 0.501). The loss trained this time (0.196 -> 0.125, M2's stayed flat) and the offset head recovered most of R2psi (0.955, against the 0.968 label-fit bound). Consistency is non-zero for the first time (0.074; Z_axis +0.52) but kappa -5.0, volume -10.2, R_axis -2.5 and li -2.6 stay negative. MAST: 100% closed, R_geo 0.76 m, elongation 2.89, overshooting MAST's ~1.8-2.2. |
| M2c | predict the offset from the external field rather than the U-Net bottleneck: the moments of the dimensionless vacuum channel on six homogeneous Grad-Shafranov fields, the moments of its change since the previous frame (vessel eddy currents follow that change) and the scalars, through a small MLP (`--offset vacuum`) | R2psi above M2b's 0.955 on DIII-D. **Failed** (`agnostic_m2c_i00`, 8 epochs, 125 min on a shared GPU): S 0.772 with borrowed q95/betaN, psi-only 0.623, R2psi 0.953, 1-D_LCFS 0.852, consistency 0.067 (Z_axis +0.47; kappa -5.1, volume -10.4, R_axis -2.3, li -2.7). Loss 0.329 -> 0.128, level with M2b from epoch 2. A tie: where the offset comes from is not the bottleneck. The consistency terms are as negative as M2b's, so the remaining error is the current's shape near the boundary, which an envelope-wide psi loss barely weights. MAST: 100% closed, R_geo 0.77 m, elongation 2.14 (M2b 2.89), now inside MAST's ~1.8-2.2, though one seed cannot separate that from run-to-run spread |
| M2d | diagnose the consistency gap (`agnostic-diagnose`): score M2c's offset and shape errors separately, and the labels' own current ([`label_current`](../../fusioneq/agnostic.py), the Grad-Shafranov operator on the label psi) through the same vacuum + Green's step; 2-epoch loss probes adding the poloidal-field error (`--loss gradient`) or supervising the current (`--loss current`) | find what caps consistency. **Found: the vac + G J structure itself.** The scorer is offset-blind: M2c's offset error alone keeps consistency 1.000 and D_LCFS 0, so all of M2c's consistency loss is shape. The labels' own current scores consistency 0.056 (R_axis -13); with the calibration refitted for it (plasma/vacuum ratio 1.06, not the ellipse's 0.90) and a label-fitted six-field vacuum correction per frame, R2psi reaches 0.984 yet consistency only 0.37, with kappa -5.4 and volume -3.8. No current this pipeline can express reaches the PCA-MLP's ~0.9. Loss probes (2 epochs, consistency; control 0.032): gradient 0.000, current 0.078, none moving kappa or volume. MAST elongation across the three probes is 1.88-2.40, so M2c's 2.14 was run-to-run spread |
| M3a | a learned flux correction: a second U-Net output channel, resampled to the machine grid and added in the offset's units, psi = s (vac + r G J) + c + s (offset + correction) mu0 \|Ip\| R_c (`--correction`) | DIII-D consistency well above the vac + G J ceiling of 0.37. **Passed at 2 epochs** (`agnostic_m3a_e2_i00`): S 0.906 with borrowed q95/betaN (control 0.757), R2psi 0.980, 1-D_LCFS 0.953, consistency 0.610 (control 0.032), every term positive for the first time (R_axis 0.56, Z_axis 0.20, kappa 0.57, tri_top 0.77, tri_bot 0.87, volume 0.65, li 0.66); loss 0.126 -> 0.024. MAST: 100% closed, R_geo 0.86 m, elongation 2.40, just above MAST's ~1.8-2.2 and inside the probes' 1.88-2.40 spread. Whether a free-form correction learned on DIII-D transfers only an official MAST score can tell. **8 epochs** (`agnostic_m3a_i00`, 57 min): S 0.940, R2psi 0.991, 1-D_LCFS 0.967, consistency 0.742 (R_axis 0.67, Z_axis 0.18, kappa 0.80, tri_top 0.89, tri_bot 0.93, volume 0.88, li 0.84); loss 0.132 -> 0.010. Z_axis is the weak term. MAST: 100% closed, R_geo 0.86 m, but elongation rose to 2.53 with training (2.40 at 2 epochs), further above MAST's ~1.8-2.2: the correction may be imprinting DIII-D shapes. **Official MAST (2026-09-30, paired with the v6 DIII-D ensemble): S 0.521 vs the physics flux's 0.575** (R2psi 0.814 vs 0.873, 1-D_LCFS 0.557 vs 0.755, consistency 0; G_ratio 0.525). The free-form correction does not transfer: it must be constrained (kept small, or trained on MAST-shaped equilibria) before another MAST slot |
| M3b | penalise the correction's energy inside the envelope, priced like an equal psi error (`--correction-weight`), 2-epoch sweep 0.01 / 0.1 / 1 | keep DIII-D consistency above 0.37 while MAST moves back toward the physics flux. **Failed as a lever.** DIII-D consistency falls with the weight (weight 0: 0.610; 0.01: 0.554; 0.1: 0.439; 1: 0.198, kappa -1.55), S 0.906 / 0.894 / 0.869 / 0.818. On MAST (all 100% closed) nothing moves monotonically: elongation 2.40 / 2.32 / 2.32 / 2.06, R_geo 0.86 / 0.85 / 1.00 / 0.85 m, R2 against the physics flux (2000 frames) 0.767 / 0.863 / 0.748 / 0.712, flux range 0.23 / 0.21 / 0.24 / 0.35 (physics 0.18). The penalty only sees DIII-D; on MAST the network shifts the work into the current instead (weight 1 doubles the flux range). Closeness to the physics flux cannot rank candidates above it either (M3a 8 epochs: 0.834 and MAST S 0.521), so without MAST labels there is no local signal to tune on |
| M3c | Grad-Shafranov-synthesised MAST equilibria (real MAST coil currents, Ip and Thomson geometry through [`GradShafranov`](../../fusioneq/equilibrium.py)) as training data and a local MAST validation set; gated on the solver's fidelity on held-out DIII-D (`gs-check`) | GS consistency clearly above the ellipse's 0. **Gate failed, not built.** The solve converges by 20 iterations to R2psi 0.776 (ellipse 0.769), consistency 0 with kappa -11.5 to -12.2 (ellipse -7.5), volume -14.3, Z_axis -17; only li improves (-0.2 vs -1.8). As M2d found, the vacuum-field model caps shape accuracy, and MAST's is built the same way, so synthetic MAST labels would teach self-consistent but wrong shapes |
| M3 | size augmentation; submit a MAST candidate | MAST S > 0.575 officially |
| M4 | q95 from psi + B0; betaN head | MAST R2_qb > 0.116 |

Rough cost: M0-M2 are about a day of implementation, and each training run is ~1 h on the A100 alongside other sweeps.

## Risks

- **Aspect-ratio extrapolation** (1.5 vs 2.8) is the main one. The explicit `R/R_c` channel and the physics `G J` step
  help, but a DIII-D-only model may still misplace MAST's current toward the centre column.
- **DIII-D's vacuum field is imperfect** (a fitted E-coil turn count, no vessel currents). The network will learn to
  compensate for DIII-D-specific errors that do not exist on MAST. Mitigation: fit the E-coil turns once and fix them
  (see the coil-turns follow-up in PR #8) rather than letting the net absorb them.
- **MAST coil conventions** were fixed once already (P-coil ampere-turns over-counted 24x); the MAST vacuum magnitude
  (peak-to-peak 0.25 vs ~0.21 expected) should be re-checked before relying on it.
- **Rules:** training uses DIII-D labels only; MAST inputs and geometry only. A MAST S of 1.0000 on the leaderboard is
  worth a rules/forum check before assuming it is reachable zero-shot.
