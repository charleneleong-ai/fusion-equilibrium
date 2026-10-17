# Why the FNO candidate lost officially (DIII-D)

Written 14-15 Oct 2026 (AEDT). Box runs 14 Oct 21:45 to 15 Oct 00:29 AEDT, CPU only. Nothing was submitted or pushed to
Hugging Face.

## Verdict

**The screen's metric is right; the submitted files were not what it screened.** The 160-shot screen scores grid shapes
predicted on the CPU in float32. 07cfd3ff, w75 (f10440d6) and every earlier DIII-D build ran on the CPU
(`CUDA_VISIBLE_DEVICES=` in `~/fusion/chain_*_build.sh`). a4b4092c (`chain_fno_build.sh`) and 3facfca7
(`chain_final16_build.sh`) were built **on the GPU**, where
[`GridPredictor._forward`](../fusioneq/grid.py#L231-L236) runs the grid models under **bf16 autocast** (`amp: true` in
every grid config). The grids carry 0.65-0.75 of the blend's flux. bf16 adds frame-to-frame noise to the flux shape,
and the magnetic axis sits where the flux is flattest, so the noise moves the O-point. That is exactly where both GPU
builds lost: R_axis and Z_axis, and nothing else of note.

The MLP members are not affected: [`NeuralPredictor.coefficients`](../fusioneq/model.py#L318-L334) calls `self.net`
without autocast. The q95 arrays of 07cfd3ff and a4b4092c agree to 1e-6, and the official R2(q95, betaN) is unchanged.

## The evidence

### Official per-scalar consistency (874 shots, scorer 3.3.1)

| build | device | screen dS (160 shots) | official dS | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
|---|---|---|---|---|---|---|---|---|---|---|
| 07cfd3ff / 688a773d (v21_27_grids3) | CPU fp32 | | (reference) | 0.980061 | 0.988840 | 0.964661 | 0.990870 | 0.989405 | 0.984058 | 0.953716 |
| a4b4092c (+ grid_fno) | **GPU bf16** | +0.000212 | **-0.000213** | **-0.00361** | **-0.00463** | +0.00043 | -0.00034 | -0.00032 | +0.00133 | -0.00017 |
| 3facfca7 (v21_16_grids3) | **GPU bf16** | +0.000027 | **-0.000354** | **-0.00447** | **-0.00566** | -0.00040 | -0.00051 | -0.00069 | +0.00010 | +0.00036 |
| f10440d6 (w75) | CPU fp32 | +0.000086 | -0.000003 | n/a | n/a | | | | | |

Both GPU builds lose 0.0036-0.0057 on each axis scalar, a loss the screen does not see for either blend. The FNO keeps
its boundary gain officially (kappa +0.00043, volume +0.00133, the two it improves on 160 shots).

### Label-free: axis jitter in the submitted test files

[`scripts/axis_jitter.py`](../scripts/axis_jitter.py) runs the official derivation (vendored `extract_lcfs_with_sign` +
`derive_frame`) on the first 60 of the 874 test shots (about 14,000 frames) of each submitted file, and reports the rms
second difference of the axis in time within a shot. Independent per-frame noise raises it; no labels are needed.
Run 14 Oct 22:35-22:48 AEDT, log `~/fusion/logs/axis_jitter_*.log`.

| file | device | R_axis jitter | Z_axis jitter | axis shift vs 07cfd3ff (rms R / Z) |
|---|---|---|---|---|
| 07cfd3ff | CPU | 6.57 mm | 8.15 mm | - |
| f10440d6 (w75) | CPU | 6.65 mm | 8.19 mm | 2.05 / 2.57 mm |
| a4b4092c (FNO) | GPU | **9.08 mm** | **10.88 mm** | 3.36 / 4.03 mm |
| 3facfca7 (v21_16_grids3) | GPU | **9.27 mm** | **10.96 mm** | 3.38 / 4.05 mm |

The two GPU builds, which are different blends, have the same jitter, and the two CPU builds have the same jitter. The
extra jitter, sqrt(9.08² - 6.57²) = 6.3 mm in R, is what independent per-frame noise of about 6.3 / sqrt(6) = 2.6 mm
gives. The official R_axis loss implies about the same: with the ground-truth spread of R_axis (0.041 m on the 160 shots),
a drop of 0.0036 in R2 is an added error of about 2.5 mm in quadrature.

### Emulated GPU build on the 160 held-out shots

[`scripts/fno_precision.py`](../scripts/fno_precision.py) re-predicts the four grids with bf16 rounding at every
convolution and linear layer (the CUDA autocast rule; `GridPredictor.predict_shape(bf16=True)` with real CPU autocast is
~13x slower for the U-Nets on this host, so it only checks the emulation on 128 frames: rms shape change real vs
emulated 2.89e-3 / 2.42e-3 for grid_v2, 2.74e-3 / 2.33e-3 grid_v2_s1, 2.81e-3 / 2.45e-3 grid_v3, 2.52e-3 / 2.44e-3
grid_fno, so the emulation slightly understates the noise). It
then rebuilds H and H + grid_fno from the cached components, rounds psi to float16 as the submission does, and scores
all 200 held-out shots frame by frame with `fusioneq/framewise.py` from the unpushed branch `feat/fusion-eq-cons-analysis` (`~/fusion/cons/` on
the box).

Run 14 Oct 22:23 to 15 Oct 00:29 AEDT, log `~/fusion/logs/fnoprec_20261014T112301Z.log`, outputs in
`~/fusion/fnodiag/out/`. Each grid's bf16 shape differs from float32 by 2.3-2.4e-3 rms (envelope units).

| config (160 never-scored shots) | S | consistency | D_LCFS | R_axis | Z_axis | kappa | volume | li |
|---|---|---|---|---|---|---|---|---|
| H, float32 psi (the screen) | 0.993914 | 0.979310 | 0.010621 | 0.98440 | 0.97484 | 0.97023 | 0.97998 | 0.95819 |
| H, float16 psi (07cfd3ff as shipped) | 0.993727 | 0.978538 | 0.010939 | 0.98315 | 0.97240 | 0.96896 | 0.97995 | 0.95824 |
| H + FNO, float16 psi | 0.993937 | 0.979460 | 0.010700 | 0.98369 | 0.97488 | 0.96996 | 0.98227 | 0.95826 |
| H, bf16 grids, float16 psi (H built on the GPU) | 0.993376 | 0.976880 | 0.011138 | 0.97897 | 0.96595 | 0.96857 | 0.97994 | 0.95818 |
| H + FNO, bf16 grids, float16 psi (a4b4092c as shipped) | 0.993627 | 0.977987 | 0.010846 | 0.97973 | 0.96923 | 0.96960 | 0.98225 | 0.95826 |

dS against the build it was compared with officially (x1e-6):

| comparison | 160 | spare0 | spare1 | spare2 | validation | scoring | official |
|---|---|---|---|---|---|---|---|
| FNO vs H, both float32 (the screen) | +212 | +157 | +83 | +338 | +251 | +120 | |
| FNO vs H, both float16 psi | +210 | +166 | +32 | +355 | +259 | +122 | |
| **FNO GPU vs H CPU (as shipped)** | **-100** | -175 | -264 | +65 | -73 | -162 | **-213** (a4b4092c) |
| H GPU vs H CPU | **-351** | | | | | | **-354** (3facfca7, a blend the screen puts +27 above H) |
| FNO vs H, both on the GPU | +252 | +195 | +109 | +378 | +306 | +147 | |

With the build path emulated, the screen reproduces both official losses: the FNO candidate turns negative in four of
five groups, and a GPU build of H itself loses 0.000351, against 3facfca7's official -0.000354. The loss sits in the axis
scalars, as officially: FNO as shipped against 07cfd3ff as shipped moves R_axis by -0.0034 and Z_axis by -0.0032
(official -0.0036 and -0.0046), while kappa (+0.0006) and volume (+0.0023) keep the FNO's gain. Built the same way on
both sides, the FNO still wins (+252, all five groups), so the FNO itself was never the problem.

The float16 cast of the submitted psi also costs H 0.000187 on 160 shots, again mostly the axis (R_axis -0.0013, Z_axis
-0.0024) and D_LCFS. It is the same for every submission so far, so it does not change a comparison, but it is a lever
(below).

## Q1: which scalars, locally (float32 screen, 160 never-scored shots)

H is 07cfd3ff's blend; FNO is a4b4092c's blend (grid_fno as a fourth grid, grids at 0.70). Frame scores from the 12 Oct
consistency run (`~/fusion/cons/fs/H.pkl`, `H_fno.pkl`). No derivation failures in either.

| | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li | consistency | S |
|---|---|---|---|---|---|---|---|---|---|
| H | 0.98440 | 0.97484 | 0.97023 | 0.99399 | 0.99353 | 0.97998 | 0.95819 | 0.979310 | 0.993914 |
| FNO | 0.98495 | 0.97707 | 0.97124 | 0.99415 | 0.99359 | 0.98233 | 0.95839 | 0.980246 | 0.994126 |
| FNO - H | +0.00055 | +0.00223 | +0.00101 | +0.00016 | +0.00005 | +0.00235 | +0.00020 | +0.000935 | +0.000212 |

Per 40-shot group, FNO - H (x1e-5; S in x1e-6):

| group | S | R_axis | Z_axis | kappa | tri_top | tri_bot | volume | li |
|---|---|---|---|---|---|---|---|---|
| spare0 | +157 | +37 | +20 | +149 | +21 | +9 | +193 | +42 |
| spare1 | +83 | +104 | +27 | -73 | +17 | +9 | +22 | +150 |
| spare2 | +338 | +44 | +380 | +198 | +37 | +2 | +622 | -287 |
| validation | +251 | +49 | +455 | +93 | -2 | +5 | +54 | +180 |
| scoring | +120 | +38 | -51 | +186 | +187 | +60 | +29 | -116 |

In float32 the FNO does not lose on any scalar pooled over 160 shots. It loses li on spare2 and scoring, kappa on spare1
and Z_axis on scoring, all small. Its axis error is no worse than H's: R_axis rms 5.08 mm vs 5.17 mm, Z_axis 5.34 vs
5.59 mm; the float32 FNO blend moves the axis by only 0.63 / 0.83 mm rms against H.

## Q2: does the screen compute consistency like the official scorer?

Yes, as far as can be checked. [`Scorer`](../fusioneq/score.py) drives the vendored, unmodified scorer modules
(`derive_frame`, `extract_lcfs_with_sign`, `finalize_machine`): the same scalar extraction, pooled R2 over all frames of
all shots, failed derivations replaced by the ground-truth mean, each scalar clipped at 0. The vendored copy is
`SCORING_VERSION = "3.2.0"`, and so is the public starter's `main` (fetched 14 Oct); 3.3.1 is not published, so its
differences cannot be diffed. Three things say they do not matter here:

- **Levels match.** H on 160 shots against 07cfd3ff on 874: R_axis 0.9844 / 0.9801, Z_axis 0.9748 / 0.9888, kappa
  0.9702 / 0.9647, tri_top 0.9940 / 0.9909, tri_bot 0.9935 / 0.9894, volume 0.9800 / 0.9841, li 0.9582 / 0.9537;
  consistency 0.9793 / 0.9788. li and kappa are the two weakest on both.
- **CPU builds transfer.** Every CPU-built DIII-D change moved the official score the same way as the screen
  (d3076dea +0.000334, 9c2c5fab +0.000249, 07cfd3ff +0.000158, at roughly half the screen's size; w75 +0.000086 locally
  is within noise of the official -0.000003).
- **The sign flips only for GPU builds**, and the label-free jitter shows the submitted files differ from the screened
  blends in exactly the scalars that lost.

So neither the metric nor the sample is the main problem. The 874 test shots also look like the 160: |Ip| median
990 kA vs 987 kA (p5-p95 556-1530 vs 558-1580), mean |z| of the coil currents against training 0.758 vs 0.720,
frames with any coil beyond 4 sigma 0.74% vs 0.47%, median 240 vs 241 frames per shot.

## Q3: FNO artefacts

In float32 there is no sign of an FNO artefact that hurts a derived scalar: the axis, li and triangularity of the FNO
blend are at least as good as H's on 160 shots, with no derivation failures. Alone, `grid_fno` places the axis about as well as
grid_v3 (rms error R 5.53 vs 5.73 mm, Z 5.53 vs 5.00 mm), with a heavier Z tail (p99 18.9 vs 14.9 mm) and a lower li
(0.9500 vs 0.9561); inside the blend these average out. The loss
officially is a build-precision effect shared with a blend that has no FNO at all (3facfca7).

## Recommendation

1. **Build in float32.** This branch makes [`GridPredictor.predict_shape`](../fusioneq/grid.py#L329-L343) float32 on
   every device; `bf16=True` reproduces the old GPU build for tests and diagnosis. Training is unchanged (autocast stays
   on in the GPU training loop). The Final rebuild runs on an A100, so without this fix every blend with grids,
   07cfd3ff included, would ship the axis loss there (about -0.00035 S, as 3facfca7 did).
2. **Screen the build, not the blend.** Two checks close the gap:
   - **Axis jitter gate**, label-free, on the built file before any submission:
     `python scripts/axis_jitter.py 60 <07cfd3ff file> <candidate file>`. A float32 build sits at 07cfd3ff's
     6.57 / 8.15 mm; a bf16 build sits near 9.1 / 10.9 mm.
   - **Calibration set**, from [`results/codabench.json`](../results/codabench.json): CPU builds with a known official
     delta (d3076dea +0.000334, 9c2c5fab +0.000249, 07cfd3ff +0.000158, w75 -0.000003) and the two GPU builds
     (a4b4092c -0.000213, 3facfca7 -0.000354). The float32 screen gets the CPU signs right; with the build emulated
     ([`scripts/fno_precision.py`](../scripts/fno_precision.py)) it now gets both GPU losses right too, which is what
     validates the diagnosis. w75's +0.000086 against -0.000003 remains a miss of the noise kind, so keep the +1e-4,
     4-of-5-groups bar. The screen's gains have run at about half their official size.
   - The **Final cuts** (base B15 vs B7, dropping grid_v2 or grid_v2_s1, fewer epochs) can be ranked by the float32
     screen as long as every candidate is then built in float32; 3facfca7's -0.000354 is a build fault and says nothing
     about v21_16_grids3 itself (float32 screen +0.000027).
3. **Development submissions before 18 Oct 11:00 AEDT.** Both are single changes with a predicted sign:
   - **a4b4092c's blend rebuilt in float32** ([`configs/blends/v21_27_grids3_fno.yaml`](../configs/blends/v21_27_grids3_fno.yaml),
     on the CPU in ~3 h, or on the GPU with this branch; jitter-gated). It differs from a4b4092c only in precision, so
     it tests the diagnosis directly. Expected: R_axis / Z_axis back to 07cfd3ff's level and S above 0.993927 (screen
     +0.000210 with float16 psi, so about +0.0001 at the usual transfer ratio). If it lands, the FNO grid is back in the
     Final candidates.
   - **float32 psi** for DIII-D, one change against the best build: +0.000187 on 160 shots for H, mostly axis. The
     validator accepts any float dtype and the starter only recommends float16 ("costs ~0.1% of score"); the cost is a
     2.7 GB DIII-D file instead of 1.4 GB against the 15 GB Codabench quota. Worth checking the rules allow it before
     spending a submission.
4. **No region blend needed.** Keeping the FNO only outside the core is unnecessary: in float32 the FNO places the axis
   as well as H, and the official axis loss is shared by a blend with no FNO.

## Not established

- The bf16 effect is emulated on the CPU (bf16 rounding at every convolution and linear layer), not reproduced on the
  GPU (CPU-only brief); against real CPU autocast on 128 frames it understates the shape noise by 3-16%.
- Scorer 3.3.1 is not public, so equality with 3.2.0's consistency path is inferred from matching levels, not diffed.
- The jitter numbers use 60 of 874 test shots.
- No official result yet for a float32 rebuild of a GPU-built blend; recommendation 3 is that test.
- The float32-psi gain is measured on 160 shots only, and whether the Final accepts float32 files is not checked.
