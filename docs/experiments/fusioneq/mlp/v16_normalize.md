# `v16_normalize` / `v17_normalize_more` — a shape-only flux target

**Schedules:** [`v16_normalize.yaml`](../../../../configs/schedules/v16_normalize.yaml),
[`v17_normalize_more.yaml`](../../../../configs/schedules/v17_normalize_more.yaml)
**Config:** `mlp`
**Chassis:** the v8 member config plus `--normalize`: the PCA is fitted on the flux **shape** (psi centred and scaled
inside the scorer's envelope, [`targets.py`](../../../../fusioneq/targets.py)) and the network also predicts that mean
and log spread; prediction inverts it exactly
**Iterations:** v16: 3 seed-matched trainings + their mean; v17: 3 more seeds
**Run:** 2026-10-04/05, ~30 min per member

## Hypothesis

The consistency scalars and the boundary depend on shape only, and part of the remaining error is frame-to-frame
jitter in the boundary our flux implies. Predicting shape separately from level and scale should lift consistency.

## Results

| seed | S normalized / plain | consistency normalized / plain | R2psi normalized / plain |
|---|---|---|---|
| 0 | 0.99014 / 0.99148 | 0.9666 / 0.9671 | 0.99812 / 0.99970 |
| 1 | 0.98920 / 0.99014 | 0.9633 / 0.9612 | 0.99744 / 0.99960 |
| 2 | 0.99036 / 0.99019 | 0.9647 / 0.9616 | 0.99888 / 0.99962 |
| 3-seed mean | 0.99186 / 0.99223 | 0.9724 / 0.9701 | 0.99856 / 0.99973 |
| v17 seeds 3 / 4 / 5 | 0.98910 / 0.99064 / **0.99184** | 0.9636 / 0.9681 / 0.9723 | |

Blended into the submitted predictor (v8 + v14 + 0.1 x trees) as a **hybrid**: the normalized ensemble's shape,
rescaled to the base flux's envelope mean and spread (so R2psi keeps the base's accuracy):

| hybrid blend | validation S | scoring S |
|---|---|---|
| none (`f4205360`, officially 0.992341) | 0.991786 | 0.992748 |
| 3 normalized (v16), w = 0.35 (`9e381a13`) | 0.992072 | 0.993242 |
| 6 normalized (v16 + v17), w = 0.5 | **0.992194** | **0.993408** |

## Verdict

- ✓ **The shape target lifts consistency** (+0.0023 on the 3-seed mean) but loses R2psi to the separately predicted level
  and scale, so alone it is slightly worse.
- ✓ **The hybrid keeps both:** the first change since v8 on which both held-out sets agree (+0.0003-0.0004 validation,
  +0.0005-0.0007 scoring), and six members beat three, so the family has not saturated.

## Next move

v18: six more seeds (twelve members); weight chosen on the validation shots again.
