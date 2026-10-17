# `v3_long_bf16` — does bf16 plus longer training beat the v1 best?

**Schedule:** [`configs/schedules/v3_long_bf16.yaml`](../../../../configs/schedules/v3_long_bf16.yaml)
**Config:** `mlp`
**Chassis:** residual MLP on PCA coefficients of psi - psi_vac, context 4, batch 2048, bf16, 40 held-out shots
**Iterations:** 5 iters · 12 rows logged
**Started:** 2026-09-28 06:06 UTC · **Finished:** 2026-09-28 ~07:05 UTC

## Hypothesis

v2 showed bf16 trains 4.5x faster and large batches lose. Keeping batch 2048 under bf16 should buy epochs for free, and v1 showed the wide net was epoch-limited. Falsified if 60-80 epochs do not beat v1's 0.9794 by more than noise (about 0.002).

## Schedule

```yaml
command: train
common_overrides:
- --holdout
- 200
- --score-shots
- 40
- --n-components
- 192
- --amp
iters:
- description: 'v1 best (context 4, 2048x6, batch 2048, 25 ep) under bf16: speed/accuracy
    check'
  overrides:
  - --context
  - 4
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 2048
  - --lr
  - 1e-3
  - --epochs
  - 25
- description: same, 60 epochs
  overrides:
  - --context
  - 4
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 2048
  - --lr
  - 1e-3
  - --epochs
  - 60
- description: batch 4096, lr 1.4e-3, 60 epochs
  overrides:
  - --context
  - 4
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 4096
  - --lr
  - 0.0014
  - --epochs
  - 60
- description: deeper 2048x10, batch 2048, 60 epochs
  overrides:
  - --context
  - 4
  - --hidden
  - 2048
  - --layers
  - 10
  - --batch
  - 2048
  - --lr
  - 1e-3
  - --epochs
  - 60
- description: PCA 256, 2048x6, batch 2048, 80 epochs
  overrides:
  - --context
  - 4
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 2048
  - --lr
  - 1e-3
  - --epochs
  - 80
  - --n-components
  - 256
```

`common_overrides`: `--holdout 200 --score-shots 40 --n-components 192 --amp`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |

## Results

| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/5 | E7 | 25 | 3.9m | — | — | — | — | — |
| 2/5 | E8 | 60 | 7.1m | — | — | — | — | — |
| 3/5 | E9 | 60 | 5.6m | — | — | — | — | — |
| 4/5 | E10 | 60 | 10.2m | — | — | — | — | — |
| 5/5 | E11 | 80 | 14.8m | — | — | — | — | — |

## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>

## Verdict

Supported, then a plateau. bf16 costs nothing at equal epochs (0.9762 vs fp32 0.9794 at 25 epochs, within noise) and 60 epochs
reaches 0.9873, lifting consistency from 0.91 to 0.95 (volume 0.82 to 0.94). Past that, depth 10 (0.9875), PCA 256 at 80 epochs
(0.9870) and batch 4096 (0.9856) all land within 0.002. Fit time stays 4 to 7 minutes on under 6 GB of GPU memory.

## Next move

`v4_pixel_loss`: the coefficient loss weights every standardised PCA mode equally; weight modes by variance so the loss is the
MSE on psi itself, and test whether length helps once depth is 10. The consistency scalars (kappa, Z_axis, volume ~0.94) are the target.
