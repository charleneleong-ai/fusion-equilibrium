# `v2_batch` — can larger batches fill the idle GPU without costing accuracy?

**Schedule:** [`configs/schedules/v2_batch.yaml`](../../../../configs/schedules/v2_batch.yaml)
**Config:** `mlp`
**Chassis:** residual MLP 2048x6 on PCA 192 of psi - psi_vac, context 2, 40 held-out shots
**Iterations:** 5 iters · 7 rows logged
**Started:** 2026-09-28 05:22 UTC · **Stopped:** 2026-09-28 06:06 UTC after 2 of 5 iters (cut short: large-batch regression was clear)

## Hypothesis

v1 runs used under 5 GB of 80 GB at 40-65% utilisation. Scaling batch 4-8x with lr scaled ~sqrt(batch) should fill the GPU and cut wall clock at equal accuracy. Falsified if S drops more than 0.005 below v1's best (0.9794).

## Schedule

```yaml
command: train
common_overrides:
- --holdout
- 200
- --score-shots
- 40
- --context
- 2
- --n-components
- 192
iters:
- description: wide 2048x6, batch 8192, lr 2e-3, 25 ep (v1 best at 4x batch)
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 8192
  - --lr
  - 2e-3
  - --epochs
  - 25
- description: same, bf16 autocast
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 8192
  - --lr
  - 2e-3
  - --epochs
  - 25
  - --amp
- description: batch 16384, lr 3e-3, 40 ep
  overrides:
  - --hidden
  - 2048
  - --layers
  - 6
  - --batch
  - 16384
  - --lr
  - 3e-3
  - --epochs
  - 40
- description: deeper 2048x10, batch 8192, 40 ep
  overrides:
  - --hidden
  - 2048
  - --layers
  - 10
  - --batch
  - 8192
  - --lr
  - 2e-3
  - --epochs
  - 40
- description: wider 4096x6, batch 8192, 40 ep, PCA 256
  overrides:
  - --hidden
  - 4096
  - --layers
  - 6
  - --batch
  - 8192
  - --lr
  - 2e-3
  - --epochs
  - 40
  - --n-components
  - 256
```

`common_overrides`: `--holdout 200 --score-shots 40 --context 2 --n-components 192`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |

## Results

| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/5 | E2 | 15 | 2.8m | — | — | — | — | — |
| 2/5 | E3 | 15 | 6.8m | — | — | — | — | — |
| 3/5 | E4 | 25 | 8.4m | — | — | — | — | — |
| 4/5 | E5 | 25 | 14.2m | — | — | — | — | — |
| 5/5 | E6 | 25 | 3.3m | — | — | — | — | — |

## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>

## Verdict

Falsified. Batch 8192 at lr 2e-3 fell to S = 0.684 in fp32 (14 min fit at 97% utilisation) and 0.915 under bf16 (3 min fit).
The third iter (batch 16k) was at train loss 3.8 at epoch 37/40 when stopped. Larger batches cut the number of optimiser
steps 4-8x and the sqrt lr scaling did not compensate. bf16 was the one clear win: 4.5x faster fit and far better than fp32
at the same settings.

## Next move

`v3_long_bf16`: bf16 everywhere, back to batch 2048-4096 at lr ~1e-3, and spend the saved time on 60-80 epochs, depth and PCA rank.
