# `v18_normalize_six` — twelve shape-target members

**Schedule:** [`v18_normalize_six.yaml`](../../../../configs/schedules/v18_normalize_six.yaml)
**Config:** `mlp`
**Chassis:** the [`v16_normalize`](v16_normalize.md) member (v8 config plus `--normalize`), seeds 6-11
**Iterations:** 6 trainings
**Run:** 2026-10-05/06, 29-44 min per member

## Hypothesis

Six shape-target members beat three in the hybrid and moved the best weight up (0.35 -> 0.5), so the family had not
saturated; twelve members should lift it again.

## Results

| seed | S | consistency | R2psi | D_LCFS |
|---|---|---|---|---|
| 6 | 0.99031 | 0.9653 | 0.99870 | 0.01347 |
| 7 | **0.99180** | 0.9712 | 0.99921 | 0.01281 |
| 8 | 0.99104 | 0.9682 | 0.99903 | 0.01289 |
| 9 | 0.99111 | 0.9699 | 0.99852 | 0.01246 |
| 10 | 0.98968 | 0.9681 | 0.99640 | 0.01257 |
| 11 | 0.99023 | 0.9669 | 0.99825 | 0.01346 |

In the hybrid (base = v8 + v14 + 0.1 x trees, normalized shape rescaled to the base's envelope mean and spread):

| hybrid blend | validation S | scoring S | official DIII-D |
|---|---|---|---|
| 6 members (v16 + v17), w = 0.5 | 0.992194 | 0.993408 | 0.99283 |
| 12 members, w = 0.5 | 0.992376 | 0.993505 | |
| **12 members, w = 0.65** (`0c9a8ac9`) | **0.992421** | **0.993566** | **0.992904** |
| 12 members, w = 0.8 | 0.992361 | 0.993545 | |

## Verdict

- ✓ **Twelve members beat six** on both held-out sets (+0.00023 validation, +0.00016 scoring) and the best weight rises
  again (0.5 -> 0.65). Officially +0.00007, about a third of the validation gain.
- Single members stay in the v16/v17 range (0.9897-0.9918); the gain is ensemble diversity, not better members.

## Next move

v19: vary the member itself (PCA size, context) for more diversity; v20: shape target plus ramp weighting.
