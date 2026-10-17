# `v19_normalize_variants` / `v20_normalize_ramp` — diverse shape-target members

**Schedules:** [`v19_normalize_variants.yaml`](../../../../configs/schedules/v19_normalize_variants.yaml),
[`v20_normalize_ramp.yaml`](../../../../configs/schedules/v20_normalize_ramp.yaml)
**Config:** `mlp`
**Chassis:** the [`v16_normalize`](v16_normalize.md) member; v19 varies the PCA size (128 / 256 vs 192) and the context
(6 / 2 vs 4), v20 adds `--ramp-weight 2.0` ([`v14_ramp`](v14_ramp.md))
**Iterations:** v19: 6 trainings (seeds 20/21); v20: 3 trainings (seeds 30-32)
**Run:** 2026-10-06, 19-46 min per member

## Hypothesis

The twelve-member hybrid's gain is diversity, not better members ([`v18`](v18_normalize_six.md)). Members that differ in
representation (PCA size, context) or in what they weight (ramp frames, which made plain members diverse in v14) should
add more than more seeds of the same recipe.

## Results

| member | S | consistency | R2psi |
|---|---|---|---|
| v19 PCA 128, seed 20 / 21 | 0.99103 / 0.99127 | 0.9708 / 0.9712 | 0.99791 / 0.99833 |
| v19 PCA 256, seed 20 / 21 | 0.98996 / **0.99166** | 0.9634 / 0.9718 | 0.99902 / 0.99869 |
| v19 context 6 / context 2, seed 20 | 0.99023 / 0.99126 | 0.9693 / 0.9701 | 0.99699 / 0.99854 |
| v20 ramp 2, seed 30 / 31 / 32 | 0.99046 / 0.99148 / 0.99010 | 0.9661 / 0.9709 / 0.9639 | 0.99863 / 0.99896 / 0.99907 |

As extra hybrid members (base = v8 + v14 + 0.1 x trees, best hybrid weight on validation):

| hybrid members | validation S | scoring S |
|---|---|---|
| 12 (`0c9a8ac9`, w = 0.65) | 0.992422 | 0.993568 |
| 12 + v20 (w = 0.65) | 0.992491 | 0.993583 |
| 12 + v19 (w = 0.75) | 0.992556 | 0.993735 |
| 21 = 12 + v19 + v20 (w = 0.65) | 0.992590 | 0.993691 |

Two further additions were tested on the same protocol:

- **Trees on the shape target:** XGBoost on the top-32 shape-PCA coefficients of `v16_normalize_i00` (its network for the
  rest), rescaled like the hybrid and blended at weight w. On the 12-member blend, w = 0.05 gives +0.00004 validation and
  +0.00004 scoring.
- **Joint re-tune** of the old tree weight (0 / 0.1 / 0.2) and hybrid weight (0.5 / 0.65 / 0.8): the submitted 0.1 / 0.65
  is already the validation optimum. No gain.

The combined blend, 21 members + shape trees:

| hybrid w | shape trees w | validation S | scoring S |
|---|---|---|---|
| 0.65 | 0 | 0.992579 | 0.993694 |
| 0.65 | 0.05 | 0.992619 | 0.993747 |
| **0.65** | **0.1** | **0.992633** | **0.993751** |
| 0.75 | 0.1 | 0.992624 | 0.993778 |

Submitted as `7a56d52a` (psi = 0.25 x base + 0.65 x hybrid + 0.1 x shape trees; MAST as `0c9a8ac9`): **official DIII-D
0.993078** against 0.992904, +0.00017, about 80% of the validation gain. Consistency 0.97400 -> 0.97481, D_LCFS
0.01156 -> 0.01144.

## Verdict

- ✓ **Member diversity pays:** six v19 variants add +0.00013 validation and +0.00017 scoring, and each pair (PCA 128,
  PCA 256, context) helps on both sets on its own (+0.00004 to +0.00008 validation).
- ✓ **Ramp-weighted shape members add on top** (+0.00003 validation with v19), though less than the v19 variants.
- ✓ **Shape trees add +0.00004** at w = 0.1 on the 21-member blend; trees remain the most diverse member type.
- ✓ The best official transfer so far (+0.00017). The 12 -> 18 -> 21 validation curve (0.992422 -> 0.992556 -> 0.992590)
  has not flattened.

## Next move

v21: six more plain shape-target seeds (27 members), tested against the submitted 21. The build and the shape-tree
blend still live in scratch scripts; port them into the repo so a re-score from source reproduces `7a56d52a`.
