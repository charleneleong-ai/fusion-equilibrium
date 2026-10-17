# `v12_temporal` — does a model over the whole shot beat the per-frame MLP?

**Schedule:** [`configs/schedules/v12_temporal.yaml`](../../../../configs/schedules/v12_temporal.yaml)
**Config:** `mlp`
**Chassis:** the v5 per-frame MLP (2048x10, context 4, PCA 192, mix loss, 120 epochs) followed by 4
[`TemporalBlock`](../../../../fusioneq/model.py)s: depthwise dilated convolutions in both directions over the shot
(+-30 frames), trained on 64-frame windows within shots and predicting one shot at a time
**Iterations:** 4 iters (3 trainings, 1 ensemble)
**Run:** 2026-10-01, about 1.8 h on the A100 (~35 min per member)

## Hypothesis

The PCA ceiling is 0.9988 (the true flux through 192 components), so the gap to the 0.9963 leader is the model's
coefficient errors, mostly kappa and li; light smoothing along time lifted the local score by ~0.0002 and past-only
history features (v10) did not. Seeing the whole shot in both directions should lift consistency on each seed-matched
pair; falsified if the members do not rise on average and the 3-seed mean does not beat 0.99223.

## Results

| seed | temporal | MLP (same seed) | change |
|---|---|---|---|
| 0 | 0.98496 | 0.99148 (`v5_loss_mix_i03`) | -0.0065 |
| 1 | 0.98660 | 0.99014 | -0.0035 |
| 2 | 0.98478 | 0.99019 | -0.0054 |
| 3-seed mean | 0.98874 (consistency 0.9583) | 0.99223 (0.9701) | -0.0035 |

## Verdict

- ✗ **Worse on every seed, by more than seed noise (~0.0013).** Every consistency term falls (0.94-0.95 per member
  against 0.96-0.97) and R2psi slips (0.9989-0.9992 vs 0.9995).
- Likely cause: each step sees 32 windows of 64 consecutive, highly correlated frames instead of 2048 frames drawn
  from the whole training set, so far fewer independent samples per step, and the temporal blocks add capacity that
  fits within shots. Together with temporal smoothing failing to transfer officially
  ([`diii-d-diagnosis.md`](../../../diii-d-diagnosis.md)), temporal coherence is not what limits DIII-D.

## Next move

None on DIII-D; `--temporal-layers` stays off by default. Remaining effort goes to MAST (Challenge 2).
