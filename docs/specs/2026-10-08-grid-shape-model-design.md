# Grid shape model for DIII-D

## Why

Every network in the submitted blend ([`v21_27_st15_2x.yaml`](../../configs/blends/v21_27_st15_2x.yaml), official
DIII-D 0.993114) predicts PCA coefficients of the flux; none sees the 65 x 65 map's spatial structure. Gains have come
only from new model families (trees +0.00015, the shape target +0.0005); more of an existing family has not. The gap
to the leader (0.9964) is boundary shape - consistency 0.975 vs 0.987, D_LCFS 0.0114 vs 0.0075 - which is decided
inside the scorer's envelope, where a convolutional decoder can put its capacity.

## Model

- **Target:** the flux *shape* - psi centred and scaled by its own mean and spread inside the envelope
  ([`shape_stats`](../../fusioneq/targets.py)). The blend consumes only shape (it moves every shape member onto the
  base flux's level and spread with [`match_envelope`](../../fusioneq/targets.py)), so level and spread are not
  predicted.
- **Inputs:** the vacuum flux map in the same shape units, the R and Z coordinate maps, and the frame's feature vector
  (coil currents, Ip, Thomson profiles with +-4 frames of context, standardised as for the MLPs).
- **Network:** a U-Net on the grid padded to 72 x 72 (three downsamplings to 9 x 9), the feature vector entering
  through an MLP that scales and shifts every level's channels (FiLM); output cropped back to 65 x 65.
- **Loss:** pixel MSE in shape units, weighted up inside the envelope (boundary and consistency live there).
- **Training:** a stride of the training frames held in RAM as float16, random batches, AdamW with one-cycle LR, bf16
  autocast; early stopping on the validation shots' envelope loss.

## Evaluation

The same 160-shot screen as the blend weights: cache the grid model's held-out shape once, then score the submitted
blend with it added as a shape component (and replacing parts of the hybrid), per 40-shot group. Promotion needs a
gain in most groups, not one pathological shot.

## Risks

- A grid model may reconstruct flux less smoothly than a low-rank PCA basis; as a blend component that matters less
  than its shape errors being different from the PCA members'.
- GPU time: the A100 is shared with another project; the model is sized to train in about an hour on the full GPU.

## Outcome (2026-10-11)

| model | data | channels / epochs | validation envelope loss | shape RMSE vs truth (validation) |
|---|---|---|---|---|
| grid_v1 | every 2nd frame | 32 / 10 | 0.00105 | 0.00254 (27-member PCA hybrid: 0.00239) |
| grid_v2 | all frames | 48 / 15 | 0.00075 | |
| grid_v2_s1 | all frames, seed 1 | 48 / 15 | 0.00065 | |
| grid_v3 | all frames | 64 / 20 | 0.00064 | |
| grid_v2_gs | all frames, Grad-Shafranov loss weight 50 | 48 / 15 | 0.0031 (closed) | |

As a blend component (weight taken out of the hybrid's 0.75), chosen on the 160 never-scored held-out shots:

| submission | grid | official DIII-D |
|---|---|---|
| `12068a0d` (no grid) | - | 0.993114 |
| `93605aab` ([`v21_27_full_grid.yaml`](../../configs/blends/v21_27_full_grid.yaml), all-frame trees) | grid_v1 at 0.1 | 0.993186 |
| `d3076dea` ([`v21_27_full_grid_v2.yaml`](../../configs/blends/v21_27_full_grid_v2.yaml)) | grid_v2 at 0.5 | 0.99352 |
| `9c2c5fab` ([`v21_27_grids2.yaml`](../../configs/blends/v21_27_grids2.yaml)) | grid_v2 + grid_v2_s1 averaged at 0.6 | 0.993769 |
| `07cfd3ff` ([`v21_27_grids3.yaml`](../../configs/blends/v21_27_grids3.yaml)) | + grid_v3, three grids at 0.65 | **0.993927** |

grid_v2's best weight is 0.5 (160 shots: 0.3 +0.00071, 0.5 +0.00092, 0.6 +0.00091, replacing the hybrid entirely
+0.00080 over `93605aab`): the grid and the PCA hybrid complement each other. The gain is consistency (0.97537 ->
0.97701); D_LCFS barely moves. A single new model family moved the score more than any blend tuning since the shape
target.

More grid models kept paying (+0.00025 for a second seed, +0.00016 for a 64-channel model) while their blend weight
rose; a Grad-Shafranov residual loss at weight 50 traded flux accuracy for force balance and lost everywhere. The whole
path is in [`journey.md`](../journey.md).
