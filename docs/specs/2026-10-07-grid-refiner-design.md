# Grid refiner for the DIII-D blend

## Why

Every model in the submitted blend ([`7a56d52a.yaml`](../../configs/blends/7a56d52a.yaml), official DIII-D 0.993078)
predicts PCA coefficients; none sees the 65 x 65 flux map's spatial structure. What has transferred officially is a
new model family (trees +0.00015, the shape target +0.0005); more members of an existing family has not (21 -> 27
members: +0.00007 locally, -0.000004 officially). About 70% of the remaining loss is consistency
([diagnosis](../diii-d-diagnosis.md)), concentrated in ramp-up / ramp-down frames, where the boundary and axis are
decided inside the envelope.

## Data

The blend's members fit their training frames, so errors there are not the errors on unseen shots. The 200 held-out
shots give honest ones without retraining any member:

| split | shots | frames | use |
|---|---|---|---|
| spare (held out, never scored or chosen on) | 120 | 27.6k | train the refiner |
| validation | 40 | 8.8k | early stopping and every choice |
| scoring | 40 | 8.2k | report only |

The blend's psi and scalars on all held-out frames are cached once (`CACHE_ROOT/refiner/held_*.npy`).

## Plan

1. **Probe (go / no-go, ~1 h).** Ridge from the blend's own PCA coefficients and the network inputs to the PCA
   coefficients of its error, fitted on the spare shots, applied with a shrink factor chosen on validation. A linear
   corrector that cannot move validation says the error is not predictable from what the blend sees; stop there.
2. **Refiner.** A small U-Net: input the blend's psi in shape units (centred / scaled in the envelope) and the vacuum
   flux; output a correction in the same units, added back at the blend's level and spread. Loss weighted to the
   envelope. Heavy regularisation (dropout, weight decay, flips are not physical - no augmentation), early stopping on
   validation.
3. **Promotion bar.** At least +0.0001 on validation with the scoring shots agreeing; below that, local gains have
   not transferred.

## Risks

- 120 shots is small for a conv net; the probe bounds what is learnable before building it.
- The refiner learns the blend's errors on held-out shots; the test shots are also unseen, so the error distribution
  matches, but any change to the blend means re-caching and retraining it.

## Outcome (2026-10-07): no-go at the probe

The blend's psi on all 200 held-out shots was cached (re-scoring the scoring shots: 0.993750, as built). Three probes,
fitted on the spare shots, chosen on validation:

| probe | best validation (blend 0.992639) | scoring (blend 0.993750) |
|---|---|---|
| ridge, raw flux error PCA(32) | 0.992639 (no correction wins; full strength 0.98795) | 0.993750 |
| ridge, envelope shape-error PCA(32) | 0.992658 | 0.993756 |
| XGBoost, 8 leading shape-error components | 0.992639 (validation error rises from round 0) | 0.993750 |

The raw-flux error is dominated by level, which consistency ignores; correcting it adds boundary noise (consistency
0.9766 -> 0.954 at full strength). In shape units the correction is inside the noise, and trees stop at once: the
error on unseen shots is shot-specific, not a pattern 120 shots can teach. A U-Net would fit the same noise.
