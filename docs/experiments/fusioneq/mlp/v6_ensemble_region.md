# `v6_ensemble_region` — do seed ensembles or an envelope-region loss lift the consistency scalars?

**Schedule:** [`configs/schedules/v6_ensemble_region.yaml`](../../../../configs/schedules/v6_ensemble_region.yaml)
**Config:** `mlp`
**Chassis:** residual MLP 2048x10 on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 120 epochs, mix loss, 40 held-out shots
**Iterations:** 6 iters (4 trainings, 2 ensembles)
**Started:** 2026-09-29 05:28 UTC · **Finished:** 2026-09-29 08:15 UTC

## Hypothesis

The official v5 score loses 0.0073 of its 0.0095 shortfall on Consistency (li 0.936, kappa 0.948), while R2psi is
0.9996. PCA 192 is not the limit: the true psi reconstructed through it scores S 0.9988 (consistency 0.996), so the
error is the network's. Two levers: averaging seeds cancels run-to-run noise in the axis and shape derivations, and extra
loss on psi inside the scorer's envelope (`--region-weight`, a quadratic form on the PCA coefficients) targets the
region where the LCFS and every derived scalar are measured. Falsified if neither beats the v5 best (0.99148) beyond noise.

## Pre-launch comparisons

| anchor | S | consistency | li | kappa |
|---|---|---|---|---|
| `v5_loss_mix_i03` (local) | 0.99148 | 0.9671 | 0.964 | 0.955 |
| `v5_loss_mix_i03` (official) | 0.99050 | 0.9637 | 0.936 | 0.948 |

## Results

| iter | run | S | R2psi | consistency | status |
|---|---|---|---|---|---|
| 0 | seed 1 | 0.99014 | 0.9996 | 0.9612 | DISCARD |
| 1 | seed 2 | 0.99019 | 0.9996 | 0.9616 | DISCARD |
| 2 | region loss, weight 1 | 0.99072 | 0.9996 | 0.9643 | DISCARD |
| 3 | region loss, weight 3 | 0.99032 | 0.9996 | 0.9624 | DISCARD |
| 4 | 3-seed ensemble (v5 best + i00, i01) | 0.99223 | 0.9997 | 0.9701 | KEEP |
| 5 | 5-model ensemble (+ i02, i03) | **0.99258** | 0.9997 | **0.9717** | KEEP |

The 3-seed ensemble was also scored by hand as `v6_ens3_seeds` (0.99220) and submitted: **official 0.99164, rank 29**
(consistency 0.9683), up from 0.99050.

## Verdict

- ✗ **Run-to-run noise is not 0.0002.** Seeds 1 and 2 of the v5 best config scored 0.99014 and 0.99019 against seed 0's
  0.99148, so single-model noise is about 0.0013. v5's winner was a lucky seed, and gaps below ~0.0015 between single
  runs in v3-v5 are not reliable.
- ✗ **Region loss, as a single model:** 0.99072 (weight 1) and 0.99032 (weight 3) are inside seed noise of the plain
  seeds; weight 3 is worse.
- ✓ **Ensembles:** averaging lifts every consistency scalar. Three seeds reach 0.99223 (+0.0020 over a typical seed);
  five models, adding the two region-loss runs, reach 0.99258. More members, not the region loss itself, is the lever.
- ✓ **The gain holds officially:** local 0.99220 became official 0.99164, a smaller local-to-official gap (0.0006) than the
  single model's (0.0010).

New best: `v6_ensemble_region_i05` (5 models), S = 0.99258, consistency 0.9717. Built and pushed as tomorrow's
candidate (HF `4eed27d3`, `pointer_20260929T0834Z_4eed27d3.zip`).

## Next move

Temporal smoothing of the ensemble's plasma residual was tested afterwards and hurt at every window (0.9895-0.9916):
frames are 20 ms apart and the equilibrium moves on that scale. The remaining input-side lever is the Thomson data,
squeezed from 44 + 10 chords onto 12 + 8 stations and taken from the nearest sample: v7
([`v7_thomson_hr.yaml`](../../../../configs/schedules/v7_thomson_hr.yaml)) trains three seeds on full-resolution,
time-averaged profiles and ensembles them with these five.
