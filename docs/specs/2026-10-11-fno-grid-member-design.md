# Fourier neural operator as a grid member (addendum to the grid shape model)

Addendum to [`2026-10-08-grid-shape-model-design.md`](2026-10-08-grid-shape-model-design.md); lever 5 of the
[2026-10-11 SOTA sweep](../../reports/2026-10-11-fusion-equilibrium-sota.md).

## Why

Every gain since the shape target has come from a structurally new model family (trees +0.00015; the U-Net grid
+0.00033 official over `93605aab`, +0.0009 on the 160-shot screen), while more of one family pays less each time (second U-Net seed +0.00025, 64 channels
+0.00016). A Fourier neural operator ([Li et al. 2020, arXiv:2010.08895](https://arxiv.org/abs/2010.08895)) sees the
whole map at every layer through a low-pass spectral convolution, where the U-Net builds context from local 3 x 3
kernels. Its errors should be smooth and global rather than local, so they should be partly uncorrelated with the
U-Nets' errors. Neural operators are the main recent surrogate family for free-boundary equilibria
([2608.05555](https://arxiv.org/abs/2608.05555)), though all of that work uses magnetics or synthetic data.

## Model

The FNO is a config choice of [`GridPredictor`](../../fusioneq/grid.py) (`GridConfig.arch = "fno"`, `train-grid
--arch fno`), not a parallel class. Target, inputs, FiLM conditioning, loss, normalisation, checkpointing, progress
logging and the blend's [`GridEnsemble`](../../fusioneq/grid.py) are all unchanged. Both networks share
`ConditionedGridNet`, which holds the feature MLP and the padded (vacuum shape, R, Z) maps.

- **Network (`FNONet`):** a 1 x 1 lift to `channels`, then `layers` Fourier blocks: `SpectralConv` (the lowest
  `modes` frequencies in each direction, mixed across channels by learned complex weights) plus a 1 x 1 linear path,
  group norm, and FiLM scale and shift from the features. A two-layer 1 x 1 head projects back to one channel.
- **Grid:** the same 65 -> 72 zero padding as the U-Net. The padding stands in for the non-periodic boundary (Li et
  al. pad the domain for the same reason). `modes` is capped at 36 (72 / 2) so the positive and negative frequency
  blocks never overlap.
- **Precision:** the FFT runs in float32 inside a disabled autocast (cuFFT has no bf16, and its half precision needs
  power-of-two sizes). The 1 x 1 convolutions keep bf16 autocast.

## Size

| model | params | saved activations, batch 256 (bf16 autocast, CPU count) | MPS fwd+bwd, batch 64 (fp32) |
|---|---|---|---|
| U-Net, 64 channels (grid_v3) | 9.78 M | 6.0 GB | 0.73 s, 4.4 GB |
| **FNO, width 48, 16 modes, 4 layers** | 10.19 M (complex weights counted as two reals; `numel()` gives 5.3 M) | 6.7 GB | 0.37 s, 2.7 GB |
| FNO, width 64, 16 modes, 4 layers | 17.61 M | 9.4 GB | 0.50 s, 4.1 GB |

The first run uses width 48: it has the U-Net's parameter count and activation memory, so it fits the same 14 GB
GPU gate that grid_v3 trained under (estimated peak 8-10 GB). It trains all frames for 20 epochs at batch 256, the
grid_v3 recipe.

## Gate (G2)

Cache the FNO's held-out shapes on the 160 never-scored shots. Score the current best blend
([`v21_27_grids3.yaml`](../../configs/blends/v21_27_grids3.yaml), official DIII-D 0.993927) with the FNO added to
the grid component (averaged with the three U-Nets, the grid weight re-chosen) and as a separate shape component.
G2 needs a gain in at least 3 of the 4 40-shot groups and a pooled gain above the 160-shot run-to-run noise; only
then is a submission spent, and the FNO is promoted only if that submission beats 0.993927.

**Expected gain:** +0.0001 to +0.0004 on the DIII-D score (speculative). Earlier new families gave +0.0006 and
+0.0008. Most of the gain should show in consistency (0.977 after grid_v2; the leader has 0.987). If the FNO's validation envelope
loss lands above about 2x grid_v3's 0.00064, its blend weight will be near zero, so close it.

## Risks

- Spectral truncation smooths the boundary. 16 modes on 72 points resolve features down to about 4.5 grid cells,
  which may be too coarse for the X-point region. The width-64 / more-modes variant is the follow-up if the loss
  plateaus above the U-Net's.
- The FFT path cannot use bf16 tensor cores, so FLOPs say little about speed on the A100. Expect an epoch close to
  the U-Net's (20-25 min on the shared GPU) rather than much faster.

Spectral weights start at `randn / width` (not Li et al.'s `1 / width²`): at width 48 the smaller scale left the spectral path ~450x weaker than the pointwise path at initialisation, so the network began as a near-pointwise model.
