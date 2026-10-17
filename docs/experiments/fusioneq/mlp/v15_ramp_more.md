# `v15_ramp_more` — do more ramp-weighted members keep adding diversity?

**Schedule:** [`configs/schedules/v15_ramp_more.yaml`](../../../../configs/schedules/v15_ramp_more.yaml)
**Config:** `mlp`
**Chassis:** as [`v14_ramp`](v14_ramp.md), seeds 3-5
**Iterations:** 6 trainings; 4 of them were killed by the GPU-hang triage while a candidate build held the machine and
were rerun as `v15b_ramp_more`
**Run:** 2026-10-04/05

## Results

| member | S |
|---|---|
| weight 2, seeds 3 / 4 / 5 | 0.99040 / 0.99047 / 0.99009 |
| weight 4, seeds 3 / 4 / 5 | 0.98928 / 0.98873 / 0.98936 |

| ensemble (validation shots) | without trees | with 0.1 x trees |
|---|---|---|
| v8 + v14 (15) | 0.991661 | 0.991788 |
| v8 + v14 + v15 (21) | 0.991307 | 0.991455 |

## Verdict

- ✗ **More ramp-weighted members hurt** (-0.00035 on validation; the scoring shots agree). v14's six helped partly by
  luck; more of the same pulls the mean toward the weaker members. The ramp direction stops here.
- Process note: run heavy builds on the A100 only between sweep iterations, or the triage misreads a data-loading
  stall as a hung GPU.
