# `v1_mlp` — does temporal context, the vacuum term, or capacity move the composite?

**Schedule:** [`configs/schedules/v1_mlp.yaml`](../../../../configs/schedules/v1_mlp.yaml)
**Config:** `mlp`
**Chassis:** residual MLP on PCA coefficients of psi - psi_vac, 61 input features, 1.36M train frames, 40 held-out shots
**Iterations:** 5 iters · 5 rows logged
**Started:** 2026-09-28 02:18 UTC · **Finished:** 2026-09-28 02:43 UTC (25 min, scorer cached + parallel)

## Hypothesis

Three levers from the first five runs: (1) temporal context should help because EFIT frames are 20 ms apart and inputs are noisy; (2) the analytic vacuum term should matter for a nonlinear model even though ridge ignored it; (3) the wide model wins, so capacity is not saturated. Falsified if the context or vacuum ablations land within noise of the baseline (about 0.005 S).

## Schedule

```yaml
command: train
common_overrides:
- --holdout
- 200
- --score-shots
- 40
- --epochs
- 15
- --batch
- 2048
iters:
- description: MLP, no context, vacuum residual (baseline)
  overrides:
  - --context
  - 0
- description: + temporal context 2
  overrides:
  - --context
  - 2
- description: context 2, no vacuum term (ablation)
  overrides:
  - --context
  - 2
  - --no-use-vacuum
- description: context 2, PCA 192, wide
  overrides:
  - --context
  - 2
  - --n-components
  - 192
  - --hidden
  - 2048
  - --layers
  - 6
- description: context 4, PCA 192, wide, 25 epochs
  overrides:
  - --context
  - 4
  - --n-components
  - 192
  - --hidden
  - 2048
  - --layers
  - 6
  - --epochs
  - 25
```

`common_overrides`: `--holdout 200 --score-shots 40 --epochs 15 --batch 2048`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |

## Results

| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/5 | E0 | 15 | 3.5m | — | — | — | — | — |
| 2/5 | E1 | 15 | 3.3m | — | — | — | — | — |
| 3/5 | E2 | 15 | 2.8m | — | — | — | — | — |
| 4/5 | E3 | 15 | 6.8m | — | — | — | — | — |
| 5/5 | E4 | 25 | 8.4m | — | — | — | — | — |

## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>

## Verdict

| lever | result |
|---|---|
| temporal context | no help at 15 epochs: context 2 scored 0.9723 vs 0.9769 without; context 4 with the wide model and 25 epochs gave the best row (0.9794), so context only pays once capacity and epochs are there |
| vacuum term | ablation 0.9720 vs 0.9723 with it: within noise for the MLP too, because the coil currents are already inputs and the residual PCA basis absorbs the linear part |
| capacity | wide 2048x6 on PCA 192 needs its epochs: 0.9666 at 15, 0.9778 to 0.9794 at 25 |

Consistency is the term that moves (0.88 to 0.93); Rel-L2 of the flux map is saturated near 0.99.
Every run reported peak GPU memory under 5 GB of 80 and mean utilisation 40 to 65%.

## Next move

`v2_batch`: batch 8k to 16k with lr scaled ~sqrt(batch), bf16 autocast, deeper and wider nets, 40 epochs,
keeping context 2 and PCA 192. Watch `phase_min/fit` and the GPU hints per run.
