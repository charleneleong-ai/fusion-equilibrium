# `v10_history` — does shot history lift li and kappa?

**Schedule:** [`configs/schedules/v10_history.yaml`](../../../../configs/schedules/v10_history.yaml)
**Config:** `mlp`
**Chassis:** as v9, plus `--history` ([`Frames.history`](../../../../fusioneq/cache.py): time in the shot, |Ip| over the
last 0.2 / 0.5 / 1 s, its 0.1 s ramp, the core peak log Te over the last 0.2 / 0.5 s)
**Iterations:** 4 iters (3 trainings, 1 ensemble)
**Started:** 2026-09-30 10:34 UTC · **Finished:** 2026-09-30 11:34 UTC

## Hypothesis

Consistency is ~80% of the gap to the 0.9963 leader, and its weakest terms, li and kappa, depend on the plasma
current profile, which diffuses inward over ~1 s. The model sees only +-4 frames (+-80 ms). Adding the shot's past
should lift li and kappa on each seed-matched pair; falsified if it does not.

## Schedule

```yaml
command: train
common_overrides: [--holdout, 200, --score-shots, 40, --amp, --context, 4, --batch, 2048, --lr, 1e-3, --n-components, 192, --hidden, 2048, --layers, 10, --epochs, 120, --coef-loss, mix, --history]
iters:
  - {description: "history, seed 0 (pair: v5_loss_mix_i03)", overrides: [--seed, 0]}
  - {description: "history, seed 1 (pair: v6_ensemble_region_i00)", overrides: [--seed, 1]}
  - {description: "history, seed 2 (pair: v6_ensemble_region_i01)", overrides: [--seed, 2]}
  - {description: "3-seed history ensemble (pair: v6_ens3_seeds)", command: ensemble}
```

## Results

Each history run against the same seed without history:

| seed | S without / with | consistency without / with | li without / with | kappa without / with |
|---|---|---|---|---|
| 0 | 0.99148 / 0.98761 | 0.9671 / 0.9503 | 0.964 / 0.947 | 0.955 / 0.933 |
| 1 | 0.99014 / 0.99005 | 0.9612 / 0.9612 | 0.959 / 0.964 | 0.945 / 0.950 |
| 2 | 0.99019 / 0.99009 | 0.9616 / 0.9618 | 0.952 / 0.963 | 0.943 / 0.943 |
| 3-seed mean | 0.99223 / 0.99145 | 0.9701 / 0.9673 | 0.964 / 0.966 | 0.956 / 0.951 |

## Verdict

- ✗ **History does not lift S or kappa.** Seeds 1 and 2 are flat (-0.0001 each) and seed 0 drops 0.0039; the 3-seed
  mean loses 0.0008. Seed 0's baseline (`v5_loss_mix_i03`) is the known lucky seed (v6 measured the seed spread at
  ~0.0013), which accounts for part of that drop but not a gain.
- ~ **li:** up on seeds 1 and 2 (+0.005, +0.011) and in the mean (+0.002), down on seed 0 (-0.017). A hint at most,
  inside the seed spread and too small to reach S.
- The past context appears to be largely recoverable already from the coil currents and Ip the model sees.

## Next move

With history, more members and every loss variant failing, the model family is at its plateau (local ~0.9928,
official 0.99217). Remaining options are structural: a larger or differently shaped network predicting psi directly
(e.g. a convolutional decoder instead of PCA coefficients), or a sequence model over the shot.
