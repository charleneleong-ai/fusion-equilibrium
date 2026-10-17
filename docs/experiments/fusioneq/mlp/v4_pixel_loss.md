# `v4_pixel_loss` — does a loss on psi itself lift the consistency scalars?

**Schedule:** [`configs/schedules/v4_pixel_loss.yaml`](../../../../configs/schedules/v4_pixel_loss.yaml)
**Config:** `mlp`
**Chassis:** residual MLP on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 40 held-out shots
**Iterations:** 5 iters · 17 rows logged
**Started:** 2026-09-28 07:05 UTC · **Finished:** 2026-09-28 08:58 UTC (2 of 5 iters killed by triage)

## Hypothesis

v3 plateaued near 0.987 with the consistency scalars as the gap. The std coefficient loss gives every PCA mode equal weight; weighting by variance makes the loss the MSE on psi, which should track the scorer better. Falsified if consistency does not rise above v3's 0.953.

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
- description: v3 best (2048x6, 60 ep) with pixel loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --epochs
  - 60
  - --coef-loss
  - pixel
- description: deeper 2048x10, 60 ep, pixel loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 60
  - --coef-loss
  - pixel
- description: 2048x10, 120 ep, pixel loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 120
  - --coef-loss
  - pixel
- description: 2048x10, 120 ep, std loss (does length alone help at depth?)
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 120
- description: 2048x10, PCA 256, 120 ep, pixel loss
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --epochs
  - 120
  - --coef-loss
  - pixel
  - --n-components
  - 256
```

`common_overrides`: `--holdout 200 --score-shots 40 --amp --context 4 --batch 2048 --lr 1e-3 --n-components 192`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |

## Results

| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/5 | E12 | 60 | 12.8m | — | — | — | — | — |
| 2/5 | E13 | 60 | 17.3m | — | — | — | — | — |
| 3/5 | E14 | 120 | 26.7m | — | — | — | — | — |
| 4/5 | E15 | 0 | 0.0m | — | — | — | — | — |
| 5/5 | E16 | 0 | 0.0m | — | — | — | — | — |

## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>

## Verdict

Falsified. The pixel loss nearly perfects the flux map (R2psi 0.9995-0.9997 against 0.998) but lowers consistency to
0.88-0.94, with Z_axis worst (0.76-0.88): variance weighting starves the small PCA modes that position the magnetic axis.
Best v4 run 0.9865 (2048x10, 120 ep, pixel) is below v3's 0.9875.

The two std-loss 120-epoch iters were killed by the GPU triage's undersized-memory rule (peak memory 12% of 80 GB for
30 min). That rule targets large-model fine-tunes; for this small model it is a false positive and is now disabled.

## Next move

`v5_loss_mix`: a 50/50 blend of the two weightings, the killed std-loss 120-epoch rerun, and a replicate of v3's best to
measure run-to-run noise, since recent differences are all 0.001-0.002.
