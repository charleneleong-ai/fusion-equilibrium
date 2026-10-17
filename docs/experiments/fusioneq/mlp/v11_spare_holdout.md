# `v11_spare_holdout` — do the never-scored held-out shots help?

**Schedule:** [`configs/schedules/v11_spare_holdout.yaml`](../../../../configs/schedules/v11_spare_holdout.yaml)
**Config:** `mlp`
**Chassis:** v8's nine member configs seed for seed, plus `--spare-holdout` (trains on the 160 held-out shots that are
never scored; the 40 scoring shots stay unseen)
**Iterations:** 10 iters (9 trainings, 1 ensemble), plus an 18-model mix scored by hand
**Run:** 2026-10-01, about 3.3 h on the A100 (~24 min per member)

## Hypothesis

Every model so far trained without the 200 held-out shots. Adding the 160 that are never scored is the standard
final-submission gain and keeps the local score comparable. Falsified if the 9-model mean does not beat
`v8_ensemble_seeds_i05` (0.99277) and the seed-matched members do not rise on average.

## Results

| member | with spare shots | without | change |
|---|---|---|---|
| seed 0 | 0.98947 | 0.99148 (`v5_loss_mix_i03`) | -0.0020 |
| seed 1 | 0.98968 | 0.99014 | -0.0005 |
| seed 2 | 0.98978 | 0.99019 | -0.0004 |
| region 1, seed 0 | 0.99005 | 0.99072 | -0.0007 |
| region 3, seed 0 | 0.99032 | 0.99032 | 0.0000 |
| seed 3 | 0.98855 | 0.98850 | +0.0001 |
| seed 4 | 0.98962 | 0.99069 | -0.0011 |
| region 1, seed 1 | 0.99068 | 0.99038 | +0.0003 |
| region 3, seed 1 | 0.99115 | 0.99004 | +0.0011 |
| member mean | 0.98992 | 0.99027 | -0.0004 |
| **9-model mean** | **0.99260** (consistency 0.9721) | 0.99277 (0.9727) | -0.0002 |
| 18-model mix (v8 + v11, `v11_mix18`) | **0.99283** (consistency 0.9730) | 0.99277 | +0.0001 |

## Verdict

- ✗ **More training shots do not help.** Members move -0.0004 on average, inside the ~0.0013 seed spread, and the
  9-model mean scores 0.99260 against 0.99277. The model is not data-limited at this size; 160 more shots on top of
  the training set change nothing measurable.
- ~ **Mixing the two sets** gives the best local number yet (0.99283), but +0.00006 over v8 is well inside
  shot-sample noise, and v9 already showed that more members saturate. Expected official ~0.9922, the same as v8.

## Next move

Data, members, losses and inputs are all exhausted for the PCA-MLP family. DIII-D gains need a structurally
different model; the larger lever is MAST, where consistency is ~0.009 (20% of the MAST score).
