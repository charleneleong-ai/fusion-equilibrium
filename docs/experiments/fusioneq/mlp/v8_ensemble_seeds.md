# `v8_ensemble_seeds` — does growing the ensemble to 7 and 9 members keep paying?

**Schedule:** [`configs/schedules/v8_ensemble_seeds.yaml`](../../../../configs/schedules/v8_ensemble_seeds.yaml)
**Config:** `mlp`
**Chassis:** residual MLP 2048x10 on PCA 192 of psi - psi_vac, context 4, batch 2048, bf16, 120 epochs, mix loss, 40 held-out shots
**Iterations:** 6 iters (4 trainings, 2 ensembles), plus an `ensemble-search` run
**Started:** 2026-09-30 04:12 UTC · **Finished:** 2026-09-30 05:45 UTC

## Hypothesis

The v6 5-model ensemble scored 0.99201 officially (local 0.99258), +0.0004 over the 3-seed ensemble, and consistency
is still ~80% of the gap to the 0.9963 leader. Averaging independent members is the one lever that has transferred,
and v6 found seed and region-loss members equally useful. Adding two of each (7 and 9 members) should lift
consistency again; it fails if the 9-model mean does not beat 0.99258.

Alongside, [`ensemble-search`](../../../../fusioneq/cli.py) tests whether a smarter combination (the median of all
members, or the mean of the best k) beats the plain mean. It chooses on 40 held-out shots that are never scored,
then scores its choice once on the usual 40.

## Schedule

```yaml
command: train
common_overrides: [--holdout, 200, --score-shots, 40, --amp, --context, 4, --batch, 2048, --lr, 1e-3, --n-components, 192, --hidden, 2048, --layers, 10, --epochs, 120, --coef-loss, mix]
iters:
  - {description: "v5 best config, seed 3", overrides: [--seed, 3]}
  - {description: "v5 best config, seed 4", overrides: [--seed, 4]}
  - {description: "envelope region loss, weight 1, seed 1", overrides: [--region-weight, 1.0, --seed, 1]}
  - {description: "envelope region loss, weight 3, seed 1", overrides: [--region-weight, 3.0, --seed, 1]}
  - {description: "7-model ensemble: v6's five + seeds 3, 4", command: ensemble}
  - {description: "9-model ensemble: v6's five + all four v8 members", command: ensemble}
```

## Pre-launch comparisons

| anchor | S | consistency | li | kappa |
|---|---|---|---|---|
| `v6_ensemble_region_i05` (5 models, local) | 0.99258 | 0.9717 | 0.966 | 0.958 |
| `v6_ensemble_region_i05` (official) | 0.99201 | 0.9698 | | |

## Results

| iter | run | S | R2psi | consistency | li | kappa | status |
|---|---|---|---|---|---|---|---|
| 0 | seed 3 | 0.98850 | 0.9996 | 0.9542 | 0.956 | 0.941 | DISCARD |
| 1 | seed 4 | 0.99069 | 0.9996 | 0.9647 | 0.962 | 0.950 | DISCARD |
| 2 | region loss 1, seed 1 | 0.99038 | 0.9997 | 0.9621 | 0.957 | 0.943 | DISCARD |
| 3 | region loss 3, seed 1 | 0.99004 | 0.9997 | 0.9606 | 0.951 | 0.947 | DISCARD |
| 4 | 7-model mean | 0.99268 | 0.9998 | 0.9723 | 0.967 | 0.958 | KEEP |
| 5 | 9-model mean | **0.99277** | 0.9998 | **0.9727** | 0.967 | 0.958 | KEEP |
| - | `ensemble-search` choice: mean of the top 5 | 0.99248 | 0.9997 | 0.9716 | 0.963 | 0.956 | DISCARD |

`ensemble-search` on the 40 never-scored shots: mean of all 9 0.99122, median 0.99027, top 3 0.99169, top 5 0.99177,
top 7 0.99138. Its per-member ranking there (best `v6_ensemble_region_i01` 0.99008, worst `v6_ensemble_region_i00`
0.98719) reorders the members' scoring-shot ranking.

## Verdict

- ✓ **More members still pay, but less each time:** 5 → 7 → 9 members score 0.99258 → 0.99268 → 0.99277 (+0.0002 for
  four more members), consistency 0.9717 → 0.9727. The expected official score of the 9-model mean is about 0.9922.
- ✗ **Smarter combinations do not:** the median is worst on the validation shots (0.99027), and the top-5 mean that
  won there (0.99177 vs 0.99122 for the plain mean) scores below the plain mean on the scoring shots (0.99248 vs
  0.99277). Member differences of ~0.001 are shot-sample noise, so selecting members fits that noise.
- The weakest single run (seed 3, 0.9885) still helps inside the mean.

New best: `v8_ensemble_seeds_i05` (9-model mean), S 0.99277, consistency 0.9727.

## Next move

Ensemble growth is flattening (+0.0001 per two members), and consistency still carries most of the gap. The
consistency loss ([design](../../../specs/2026-09-30-consistency-loss-design.md)) targets it directly: li and volume
linearise around the label (C1), and a fine-tune of `v5_loss_mix_i03` with that term is the next test (C3).
