# `v21_normalize_seeds` — 27 shape-target members (negative officially)

**Schedule:** [`v21_normalize_seeds.yaml`](../../../../configs/schedules/v21_normalize_seeds.yaml)
**Config:** `mlp`
**Chassis:** the v18 shape-target member, seeds 40-45
**Iterations:** 6 trainings, 18-27 min each (shared GPU)
**Run:** 2026-10-07

## Hypothesis

The 12 -> 18 -> 21 member curve had not flattened on validation (0.992422 -> 0.992556 -> 0.992590), so 27 members
should lift the submitted blend ([`7a56d52a.yaml`](../../../../configs/blends/7a56d52a.yaml)) again.

## Results

| seed | 40 | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|
| S | **0.99187** | 0.99029 | 0.99154 | 0.98985 | 0.98988 | 0.99171 |

Seeds 40 and 45 are the best single members so far. In the blend (repo [`Blend`](../../../../fusioneq/blend.py),
weights chosen on validation; hybrid 0.85 and shape trees 0.15 checked and lower):

| blend | validation | scoring | official DIII-D |
|---|---|---|---|
| 21 members, hybrid 0.65 (`7a56d52a`) | 0.992639 | 0.993750 | **0.993078** |
| 27 members, hybrid 0.75 ([`v21_27.yaml`](../../../../configs/blends/v21_27.yaml), `6525ba1d`) | 0.992713 | 0.993826 | 0.993074 |

## Verdict

- ✗ **+0.00007 on both local sets, -0.000004 officially.** More members of the same family have stopped paying, and
  local gains below ~0.0001 are inside the noise of 40 + 40 shots even when both sets agree. The two gains that did
  transfer (shape target, 21 members + shape trees) were +0.0002 or more on validation.

## Next move

No more members of this family. New directions tested the same night, all below the bar: q95 / betaN from the
shape members' heads or trees (validation +0.000055, scoring -0.000033), and a grid refiner
([spec](../../../specs/2026-10-07-grid-refiner-design.md)), whose probes found the blend's held-out error unpredictable.
