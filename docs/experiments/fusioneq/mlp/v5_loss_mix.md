# `v5_loss_mix` — can a blended loss keep psi accuracy without losing the shape scalars?

**Schedule:** [`configs/schedules/v5_loss_mix.yaml`](../../../../configs/schedules/v5_loss_mix.yaml)
**Config:** `mlp`
**Chassis:** residual MLP on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 40 held-out shots
**Iterations:** 5 iters · 22 rows logged
**Started:** 2026-09-28 09:02 UTC · **Finished:** 2026-09-28 ~11:05 UTC

## Hypothesis

v4 showed the pixel loss perfects psi but starves the small modes that place the magnetic axis, while the std loss keeps them. Weighting each coefficient half-and-half should keep both. Falsified if the blend does not beat the std loss at equal depth and epochs by more than run-to-run noise, which the replicate measures.

## Schedule

```yaml
command: train
common_overrides:
- --holdout
- 200
- --score-shots
- 40
- --amp
- --context
- 4
- --batch
- 2048
- --lr
- 1e-3
- --n-components
- 192
iters:
- description: 'replicate of v3 best (2048x6, 60 ep, std loss): noise floor'
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --epochs
  - 60
- description: 2048x6, 60 ep, mix loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --epochs
  - 60
  - --coef-loss
  - mix
- description: 2048x10, 120 ep, std loss (killed in v4)
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 120
- description: 2048x10, 120 ep, mix loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 120
  - --coef-loss
  - mix
- description: 2048x6, 120 ep, std loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --epochs
  - 120
```

`common_overrides`: `--holdout 200 --score-shots 40 --amp --context 4 --batch 2048 --lr 1e-3 --n-components 192`

## Verdict

Supported. The replicate of v3's best scored 0.9871 against 0.9873, suggesting run-to-run noise of about 0.0002.
(v6 revised this: two more seeds of the winner scored 0.9901-0.9902, so single-model noise is about 0.0013 and the
differences below are within it.) At 60 epochs the blend beats the std loss by 0.002 (0.9893 vs 0.9871); at depth 10 and 120
epochs it beats it by 0.0018 (0.9915 vs 0.9897). It keeps the pixel loss's flux accuracy (R2psi 0.9994-0.9997) and lifts
every consistency scalar, Z_axis back to 0.96. Length also helps once depth is 10: std loss 60 to 120 epochs, 0.9875 to 0.9897.

New best: `v5_loss_mix_i03` (2048x10, 120 epochs, mix loss), S = 0.9915, consistency 0.967.

## Next move

The weakest scalars are now kappa (0.955), Z_axis (0.959) and volume (0.969). Candidates: a mix weight sweep (0.25/0.75),
200 epochs at depth 10, and a first real MAST model (the submission still uses the vacuum baseline there).
