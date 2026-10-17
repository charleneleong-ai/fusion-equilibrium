# `v14_ramp` — does weighting ramp-up / ramp-down frames lift consistency?

**Schedule:** [`configs/schedules/v14_ramp.yaml`](../../../../configs/schedules/v14_ramp.yaml)
**Config:** `mlp`
**Chassis:** the v8 member config plus `--ramp-weight` ([`ramp_frames`](../../../../fusioneq/cache.py): the first or
last 10% of each shot, or |Ip| below 80% of the shot's peak; ~26% of training frames)
**Iterations:** 6 trainings (weights 2 and 4, three seeds each)
**Run:** 2026-10-04, about 2.2 h on the A100

## Hypothesis

The worst 1% of held-out frames hold 27-57% of each consistency term's error, mostly ramp phases at low current and
high li ([`diii-d-diagnosis.md`](../../../diii-d-diagnosis.md)). Weighting those frames up in the loss should lift li,
Z_axis and volume on seed-matched pairs.

## Results

| seed | unweighted | weight 2 | weight 4 |
|---|---|---|---|
| 0 | 0.99148 | 0.98988 | 0.99060 |
| 1 | 0.99014 | 0.98962 | 0.98934 |
| 2 | 0.99019 | 0.98980 | 0.98963 |
| mean | 0.99060 | 0.98977 | 0.98986 |

As ensemble members, on the never-scored validation shots: v8 + the six v14 members 0.991661 against 0.991218 for v8
(+0.00044), where six plain extra members (v9) add +0.00007.

## Verdict

- ✗ **Weaker as single models** (-0.0007 to -0.0008 on matched seeds); li / Z_axis / volume move both ways.
- ✓ **But diverse:** six of them lift the ensemble six times as much as six plain members. Officially, v8 + v14 + 0.1 x
  trees (`f4205360`) scored 0.992341 against 0.992317 without them (+0.00002): about a sixth of the validation gain.

## Next move

[`v15_ramp_more`](v15_ramp_more.md) tests whether more of them keep adding.
