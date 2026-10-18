# DIII-D finishing step

Screened 17-18 Oct 2026 (AEDT) on the A100 host, CPU for the finishing itself. **Verdict: PASS.** The recipe
[`configs/finish/finish_v1.json`](../configs/finish/finish_v1.json) adds **+440e-6 S** on the 160 never-scored
held-out shots to the Final pick ([`v21_16_grids3_fno2`](../configs/blends/v21_16_grids3_fno2.yaml), 0.994433 ->
0.994873). It gains on all five 40-shot groups, costs R2psi 9e-6, and finishes the whole DIII-D public test
set (193,359 frames) on CPU in **40 min** wall time (35 min of finishing), on a shared 16-core host at load 34.

This is a held-out screen only. No official score has been taken: past screens transferred to the official score at
roughly 0.5x, and one (the first FNO pick, [`final-budget.md`](final-budget.md)) did not transfer at all.

## What it does

The scorer derives seven consistency scalars from the submitted map (R_axis, Z_axis, kappa, tri_top, tri_bot,
volume, li). The finishing step deforms each predicted map so that the scorer reads values closer to the truth. This
follows the idea in two published top-15 pipelines (Hakami, MIT; Anulum, read as a description only). The code is our
own.

1. **Heads.** [`ScalarHeads`](../fusioneq/scalar_heads.py) is a residual MLP (1024 x 6) on the members' own inputs
   (coil currents, Ip, high-resolution Thomson, +-4 frames of context, shot history). It is trained on the scorer's
   `derive_frame` of the label flux on every DIII-D training frame (1,514,702 frames; the 200 held-out shots never
   enter). Four seeds are averaged. Training is float32 with TF32 off, about 7.5 min per seed on the A100, so about
   0.5 A100-h for four.
2. **Targets.** For each scalar, target = map reading + w (head - map reading). The weight w was chosen per scalar on
   the 40 validation shots only: R_axis 0.4, Z_axis 0, kappa 0.45, tri_top 0.7, tri_bot 0.7, volume 0.5, li 0.35.
3. **Deformation** ([`fusioneq/finish.py`](../fusioneq/finish.py)):
   - **Warp.** A smooth displacement psi(x - u(x)) built from six basis fields: two Gaussian bumps at the axis, plus a
     vertical stretch, a radial stretch and upper and lower shears inside a window around the plasma.
   - **Solve.** The six coefficients are solved by Gauss-Newton on the advected LCFS polygon. There are two passes,
     each re-reading the warped map exactly as the scorer does.
   - **Guard.** A candidate warp is dropped if the re-extracted LCFS moved more than 2x the intended displacement.
   - **li.** Then the monotone core remap from [`li_shape.CoreShaper`](../fusioneq/li_shape.py) is applied, on the
     scorer's own masked contour. It changes li and leaves every flux surface in place.
   - **Gate.** A frame keeps the finished map only if the scorer's reading of it is nearer the targets.

## Protocol (fixed before results)

- Heads are trained on training frames only.
- Every hyperparameter was chosen on the `validation` group only: head architecture, blend weights, operators and the
  final variant.
- Results are reported on the 160 never-scored shots and the five 40-shot groups exactly as
  `~/fusion/final_budget/subsets160_fno.py` defines them. The baseline is the Final pick's own float32 psi, which
  reproduces its recorded 0.994433.
- **Gate:** dS > +2e-4 on the 160 shots; a gain on at least 4 of the 5 groups; R2psi not lower by more than 2e-5; the
  step runs on the DIII-D test set on CPU in under 2 h.

## Result (dS x 1e-6 vs the Final pick)

| Variant | 160 shots | spare0 | spare1 | spare2 | validation | scoring | dR2psi (160) | dD_LCFS (160) |
|---|---|---|---|---|---|---|---|---|
| **finish_v1: m8 x4 seeds, all operators** | **+440** | +296 | +244 | +1048 | +243 | +493 | -9e-6 | +0.9e-5 |
| m8 (1 seed), all operators | +422 | +325 | +213 | +984 | +233 | +521 | -7e-6 | +4.6e-5 |
| m8, axis + li only | +251 | +86 | +70 | +791 | +68 | +127 | +1e-6 | -0.1e-5 |
| m7, all operators | +287 | +224 | +188 | +656 | +143 | +333 | -6e-6 | +8.7e-5 |
| m7, axis + li only | +172 | +63 | +76 | +526 | +43 | +72 | +1e-6 | 0 |
| m1, first operator version | +20 | +51 | -12 | +10 | +38 | +176 | -10e-6 | +28e-5 |

finish_v1 on the 160 shots: Consistency 0.98166 -> 0.98389. Each variant row was finished on all 200 held-out shots.
The pick among them was made on the validation column (+243 was the best there). The baseline S per group is in
`~/fusion/finishing/baseline_groups.json`, and each variant's per-scalar R2 is in `finish_<variant>.json` next to it.

The operator ceiling, measured on validation with the **true** scalars as targets (a perfect head), is +3.0e-3 S:

| Operator | dS |
|---|---|
| axis | +1.34e-3 |
| shape | +1.00e-3 |
| li | +0.73e-3 |
| all | +3.02e-3 |

With the oracle targets R2psi rises (the li remap moves psi toward the truth), so the operators themselves cost
nothing. The limit is the heads.

## The funnel

**Heads.** Pooled consistency R2 of the blended targets, before any deformation. The map alone scores 0.98412 on
validation and 0.98166 on the 160 shots. Selection was on validation; the 160-shot column is shown for transparency.

| Heads | Validation | 160 shots | Note |
|---|---|---|---|
| XGBoost per scalar | - | - | dropped: ran single-threaded (5 min per target); R_axis inner RMSE 7.2 mm vs the map's 4.8 mm |
| m1 MLP 512x4, lr 1e-3, every 4th frame | 0.98478 | 0.98253 | |
| m2 1024x4, 100 epochs | 0.98475 | 0.98271 | |
| m3 1024x6, dropout 0.1 | 0.98490 | 0.98287 | |
| m4 = m1 with context 8 | 0.98449 | 0.98204 | wider context hurts |
| m3 x 4 seeds | 0.98493 | 0.98317 | |
| m5 every 2nd frame, lr 1e-3 | 0.98455 | 0.98146 | training diverged mid-run (loss 0.03 -> 0.18) |
| m6 = m3 with lr 3e-4 and grad clip 1 | 0.98503 | 0.98348 | stable; li starts to help |
| m7 = m6 on every 2nd frame | 0.98531 | 0.98392 | |
| m8 = m6 on every frame, 40 epochs | 0.98578 | 0.98490 | |
| **m8 x 4 seeds** | **0.98582** | **0.98496** | finish_v1 |

The heads never beat the map on their own: m8's mean is 0.98058 on validation. They help only blended into it. Within
the blend, li, kappa and the triangularities gain most.

**Operators.** Validation dS, before the final full runs:

| Operator setting (heads m3 / m6) | Validation dS |
|---|---|
| m3, all operators, no gate | -413e-6 |
| m3, all operators, gate | +76e-6 |
| m3, axis + shape / shape only, no gate | -413e-6 / -427e-6 |
| m3, all, no gate, jump guard 2x | -166e-6 |
| m3, all, gate + jump guard 2x | +71e-6 |
| m3, all, gate + jump 1.5x + 3 passes | +53e-6 |
| m6, all, per-pass best candidate + gate + jump 2x | +91e-6 |
| m6, same with step scales 0.5 / 1 / 1.5 / 2 | +89e-6 |
| m6, axis + li | +70e-6 |

**Why the shape operators are noisy.** The scorer reads kappa and the triangularities off a contour extracted at the
X-point level, on a 2.7 cm x 5 cm grid. Near the X-point, the extracted boundary moves in grid-cell steps rather than
with the warp, so a single frame's kappa can jump by about 0.04 for a millimetre of warp.
- Ungated, this noise outweighs the heads' small corrections (kappa R2 0.9835 -> 0.9752).
- The gate, the jump guard and the per-pass "keep only if nearer" rule turn that loss into a gain.
- The axis and li operators are smooth: realised over intended is 0.97 and 1.00.

## Runtime

- **Held-out screen:** 44,638 frames in 356 s on 14 worker processes, on a shared host at load 34.
- **CLI on the DIII-D public test set, first run:** `fusioneq finish`, CPU only, took **2 h 29 min** wall time
  (2 h 26 min finishing; peak RSS 11 GB), which is over the 2 h gate. The workers were forked after torch had run
  the heads, and each started a full BLAS / OpenMP thread pool, so 14 processes oversubscribed a 16-core host already at
  load 34. The screen runs had `OMP_NUM_THREADS=1` set. `Finisher.many` now pins one thread per worker
  (`one_thread`).
- **CLI on the DIII-D public test set, after the fix:** **40 min 3 s** wall time, of which 34 min 48 s was finishing
  (96.6% of frames accepted). Peak RSS was 11 GB. Another GPU job of ours was running at the same time, and the host
  was at load 34.
- **Cost per frame:** about 4 scorer reads (the initial one, two warp passes and the final one) plus the li curve.

## Rebuild path, checked end to end

The rebuild steps were run through the CLI on a symlinked cache root, so the real cache was never written:
- `finish-targets --stride 1` took 35 min 49 s on CPU (12 workers, 1,136,026 frames derived, the rest reused from
  `cons_targets.npz`). Its labels are identical to the screen's.
- `train-heads --seed 0` took 7 min 11 s on the shared A100.
- The retrained head's held-out predictions match the screened `m8` head **bitwise**, so the rebuilt heads are the
  scored heads.
- Four seeds add about 29 min, about 0.5 A100-h, to the Final's training.

## Residual heads (18 Oct): not promoted

**Hypothesis.** A head that also sees the map's own reading, and predicts truth minus reading, would be more accurate
than the input-only heads. The gate was fixed before the results: beat finish_v1 by more than +2e-4 on the 160 shots,
gain on at least 4 of 5 groups, and add less than 1.5 A100-h. It **fails** at the target stage, so no variant was worth
finishing.

| Heads (validation-chosen weights) | Validation | 160 shots | Note |
|---|---|---|---|
| finish_v1 heads (m8 x4) | 0.98582 | 0.98496 | reference |
| r1: residual on in-sample readings, 1024x6 | 0.98481 | 0.98257 | |
| r2: residual on in-sample readings, 512x4, weight decay 1e-3 | 0.98489 | 0.98272 | beats the map alone (0.98466 vs 0.98412) |
| m8 x4 + r2, two weights per scalar | 0.98611 | 0.98522 | **+0.00029 / +0.00026 Consistency, so about +6e-5 S at full realisation, under the +2e-4 gate** |
| p1: residual on out-of-sample proxy readings | 0.98438 | - | proxy map too weak; li NaN where the proxy has no LCFS |
| m8 x4 + p1 | 0.98582 | - | p1 gets weight 0 |

**Readings used.**
- **In-sample** ([`map_train.py`](../scripts/finishing/map_train.py)): the Final pick's own psi on every 4th training
  shot, 380,309 frames. GPU inference took 34 min on the shared A100 and the CPU read took 10 min.
  - The members were trained on these frames, so the readings are far too good: in-sample mean R2 0.9908 (li 0.992),
    against 0.984 on validation (li 0.975).
  - The residual heads therefore learn corrections that are too small.
- **Out-of-sample proxy** ([`kfold_proxy.py`](../scripts/finishing/kfold_proxy.py)): two half-split MLPs (1024x6,
  40 epochs), 19 min of GPU.
  - Their held-out reading is poor (validation mean R2 0.89, volume 0.75), so it adds nothing.
  - A proxy close to the Final pick's quality would need the Final's members retrained K-fold, which does not fit the
    budget.

## Use

```bash
# rebuild, finishing included: finish-targets (CPU), 4 x train-heads (GPU, ~0.5 A100-h), then build --finish
python -m fusioneq.cli rebuild --spec configs/blends/v21_16_grids3_fno2.yaml --finish configs/finish/finish_v1.json

# or finish an existing DIII-D submission .npz (heads under results/heads_finish_v1_s{0..3})
python -m fusioneq.cli finish --spec configs/finish/finish_v1.json \
  --submission <dir>/diii_d_public_test.npz --out <out>/diii_d_public_test.npz --config diii_d_public_test
```

The screen scripts are in [`scripts/finishing/`](../scripts/finishing/). They run on the host with
`PYTHONPATH=<repo>:~/fusion/finishing` and read `~/fusion/final_budget/subsets160_fno.py`:
- `prep.py`: the Final pick's held-out psi, its readings and the labels;
- `labels_all.py`: labels on every training frame;
- `heads_mlp.py`, `blend_eval.py`: the heads and their validation-chosen blend weights;
- `oracle.py`, `diag.py`: the operator ceiling and the validation diagnostics;
- `finish_eval.py`: a variant finished on all 200 held-out shots and scored per group.

## Not established

- **Official transfer.** Only the Development leaderboard could test it, and nothing was submitted.
- **Shape gains may be partly lucky.** The kappa and triangularity gains depend on how the X-point falls on the grid.
  The axis + li variant is the smooth, low-risk fallback: +251e-6 on 160 shots, R2psi +1e-6, D_LCFS unchanged.
- **X-point placement and the LCFS ray heads** that the two published pipelines use were not tried.
