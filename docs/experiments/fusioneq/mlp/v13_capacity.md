# `v13_capacity` — do wider or longer members lift consistency?

**Schedule:** [`configs/schedules/v13_capacity.yaml`](../../../../configs/schedules/v13_capacity.yaml)
**Config:** `mlp`
**Chassis:** the v8 member config (residual MLP, 10 layers, PCA 192, context 4, mix loss) at 4096 wide and / or 240
epochs instead of 2048 / 120
**Iterations:** 5 trainings
**Run:** 2026-10-02, about 4 h on the A100 (32-45 min at 4096 x 120 or 2048 x 240, ~75 min at 4096 x 240)

## Hypothesis

Capacity was last swept with the old loss (v3 plateaued at 2048x10, 60 epochs), and our consistency (0.9705) is the
weakest among the teams at similar S (0.975-0.979). Wider or longer members should lift consistency on seed-matched
pairs; worth a submission only above ~+0.0005, since smaller local gains have proved to be noise.

## Results

| config | seed 0 | seed 1 | mean | 2048 x 120 (same seeds) |
|---|---|---|---|---|
| 4096 wide, 120 epochs | 0.98879 | 0.98773 | 0.98826 | 0.99148 / 0.99014 (0.99081) |
| 2048 wide, 240 epochs | 0.98999 | 0.98910 | 0.98955 | 0.99081 |
| 4096 wide, 240 epochs | 0.98929 | | | 0.99148 |

## Verdict

- ✗ **More capacity is worse on every seed.** Wider loses 0.0026 on the mean, longer 0.0013, both together 0.0022 on
  seed 0; consistency falls with them (0.953-0.959 against ~0.96), kappa most. The 2048 x 10, 120-epoch member is
  the best configuration found; extra capacity fits the training shots rather than the shape scalars.

## Next move

None on capacity. A second model family (gradient-boosted trees) for ensemble diversity was tested alongside, also
negative: see [`diii-d-diagnosis.md`](../../../diii-d-diagnosis.md).
