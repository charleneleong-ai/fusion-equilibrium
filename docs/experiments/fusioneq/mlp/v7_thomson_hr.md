# `v7_thomson_hr` — do full-resolution, time-averaged Thomson inputs lift li and the shape scalars?

**Schedule:** [`configs/schedules/v7_thomson_hr.yaml`](../../../../configs/schedules/v7_thomson_hr.yaml)
**Config:** `mlp`
**Chassis:** residual MLP 2048x10 on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 120 epochs, mix loss, `--thomson-hr`, 40 held-out shots
**Iterations:** 5 iters (3 trainings, 2 ensembles)
**Started:** 2026-09-29 08:16 UTC · **Finished:** 2026-09-29 ~10:20 UTC

## Hypothesis

Consistency is 76% of the remaining DIII-D loss and li (0.940 official) its weakest scalar. li follows the current
profile, which follows the Te profile shape, yet the standard inputs squeeze 44 core + 10 edge Thomson chords onto
12 + 8 stations and take the single nearest sample per 20 ms frame, while core Thomson samples every ~3 ms.
`--thomson-hr` uses one station per chord and averages each channel's samples within +-10 ms (108 Thomson features
instead of 40; cache `thomson_hr.npy` from `prepare-thomson`). Falsified if the same seeds do not beat the standard
inputs beyond the ~0.0013 seed noise, and if the 3-seed ensemble does not beat v6's 0.99223.

## Pre-launch comparisons

| anchor | S | consistency | li |
|---|---|---|---|
| standard Thomson, seed 0 (`v5_loss_mix_i03`) | 0.99148 | 0.9671 | 0.964 |
| standard Thomson, seed 1 (`v6_ensemble_region_i00`) | 0.99014 | 0.9612 | 0.959 |
| standard Thomson, seed 2 (`v6_ensemble_region_i01`) | 0.99019 | 0.9616 | 0.952 |
| 3-seed ensemble (`v6_ensemble_region_i04`) | 0.99223 | 0.9701 | – |
| 5-model ensemble (`v6_ensemble_region_i05`, held candidate) | 0.99258 | 0.9717 | – |

## Results

| iter | run | S | consistency | li | vs standard |
|---|---|---|---|---|---|
| 0 | seed 0 | 0.98820 | 0.9527 | 0.955 | -0.0033 |
| 1 | seed 1 | 0.98880 | 0.9558 | 0.937 | -0.0013 |
| 2 | seed 2 | 0.98950 | 0.9590 | 0.944 | -0.0007 |
| 3 | 3-seed ensemble | 0.99094 | 0.9648 | 0.952 | -0.0013 vs 0.99223 |
| 4 | 6-model mix (v5/v6 seeds + these) | 0.99237 | 0.9709 | 0.963 | -0.0002 vs 0.99258 |

All DISCARD.

## Verdict

- ✗ **Seed for seed, the full-resolution inputs are worse** on all three seeds (-0.0033, -0.0013, -0.0007), and li,
  the target, falls on every seed.
- ✗ **As an ensemble** they lose to the standard-input ensemble (0.99094 vs 0.99223), and as extra members they drag
  the 6-model mix just below the 5-model ensemble.
- Likely cause: with 4 context frames either side, the Thomson inputs grow from 360 to 972 features. The per-chord
  detail is mostly noise the MLP overfits, and averaging within a frame may blur real profile changes.

The held candidate stays `v6_ensemble_region_i05` (0.99258). `--thomson-hr` stays available, off by default, as a
documented negative result.

## Next move

Input-side levers on DIII-D are now exhausted short of new information: prediction smoothing (tested after v6) and
richer Thomson both hurt. The remaining DIII-D gains are more ensemble members of the standard config, and losses on
the derived quantities themselves. The larger lever is Award 2: the machine-agnostic network
([design](../../../specs/2026-09-29-machine-agnostic-network-design.md)), whose M2 run finished after this sweep.
