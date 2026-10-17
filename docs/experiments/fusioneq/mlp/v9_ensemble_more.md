# `v9_ensemble_more` — does the ensemble keep improving past nine members?

**Schedule:** [`configs/schedules/v9_ensemble_more.yaml`](../../../../configs/schedules/v9_ensemble_more.yaml)
**Config:** `mlp`
**Chassis:** residual MLP 2048x10 on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 120 epochs, mix loss, 40 held-out shots
**Iterations:** 10 iters (8 trainings, 2 ensembles)
**Started:** 2026-09-30 07:41 UTC · **Finished:** 2026-09-30 10:34 UTC

## Hypothesis

The v8 9-model mean scored 0.99217 officially (rank 26; local 0.99277), +0.00016 over the 5-model mean, and each
doubling of members had added ~0.0002 locally. Eight more members (four seeds of the base config, two of each
region-loss weight) should lift the 13- and 17-model means again; falsified if the 17-model mean does not beat
0.99277.

## Schedule

```yaml
command: train
common_overrides: [--holdout, 200, --score-shots, 40, --amp, --context, 4, --batch, 2048, --lr, 1e-3, --n-components, 192, --hidden, 2048, --layers, 10, --epochs, 120, --coef-loss, mix]
iters:
  - {description: "v5 best config, seeds 5-8", overrides: [--seed, 5..8]}
  - {description: "envelope region loss, weight 1, seeds 2 and 3", overrides: [--region-weight, 1.0, --seed, 2..3]}
  - {description: "envelope region loss, weight 3, seeds 2 and 3", overrides: [--region-weight, 3.0, --seed, 2..3]}
  - {description: "13-model mean: v8's nine + seeds 5-8", command: ensemble}
  - {description: "17-model mean: all members", command: ensemble}
```

## Pre-launch comparisons

| anchor | S | consistency | li | kappa |
|---|---|---|---|---|
| `v8_ensemble_seeds_i05` (9 models, local) | 0.99277 | 0.9727 | 0.967 | 0.958 |
| `v8_ensemble_seeds_i05` (official) | 0.99217 | 0.9705 | | |

## Results

| iter | run | S | R2psi | consistency | li | kappa | status |
|---|---|---|---|---|---|---|---|
| 0 | seed 5 | 0.99050 | 0.9995 | 0.9636 | 0.961 | 0.951 | DISCARD |
| 1 | seed 6 | 0.98850 | 0.9995 | 0.9560 | 0.960 | 0.934 | DISCARD |
| 2 | seed 7 | 0.98960 | 0.9997 | 0.9591 | 0.958 | 0.941 | DISCARD |
| 3 | seed 8 | 0.99034 | 0.9995 | 0.9625 | 0.958 | 0.946 | DISCARD |
| 4 | region loss 1, seed 2 | 0.98947 | 0.9995 | 0.9594 | 0.939 | 0.945 | DISCARD |
| 5 | region loss 1, seed 3 | 0.99016 | 0.9993 | 0.9621 | 0.957 | 0.954 | DISCARD |
| 6 | region loss 3, seed 2 | 0.98952 | 0.9998 | 0.9588 | 0.949 | 0.949 | DISCARD |
| 7 | region loss 3, seed 3 | 0.98954 | 0.9996 | 0.9588 | 0.951 | 0.940 | DISCARD |
| 8 | 13-model mean | 0.99277 | 0.9998 | 0.9727 | 0.968 | 0.957 | DISCARD |
| 9 | 17-model mean | 0.99272 | 0.9998 | 0.9723 | 0.966 | 0.957 | DISCARD |

## Verdict

- ✗ **Averaging has saturated.** 9 -> 13 -> 17 members score 0.99277 -> 0.99277 -> 0.99272; consistency 0.9727 ->
  0.9727 -> 0.9723. The members' errors are no longer independent enough for the mean to cancel them further.
- The new members sit in the same 0.9885-0.9905 band as v6/v8's, so seed-to-seed spread (~0.0013) is unchanged.

The best DIII-D model stays `v8_ensemble_seeds_i05` (9-model mean, local 0.99277, official 0.99217).

## Next move

Cheap levers are exhausted: more members (v9), envelope-region loss (v6), gradient / current / linearised-consistency
losses and full-resolution Thomson all failed, and history features (v10,
[`v10_history.md`](v10_history.md)) did not move S. The ~0.004 gap to the 0.9963 leader is mostly consistency (li,
kappa), so the next step needs a structurally different model rather than more of the same members.
