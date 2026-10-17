# Consistency loss for the DIII-D network — design

**Status:** closed, negative, 2026-09-30. **Goal:** close the DIII-D gap to the leader (official S 0.99201 vs 0.9963), about
80% of which is consistency (0.970 vs 0.988; li 0.940 and kappa 0.953 are the weakest terms).

## Why the current loss cannot see it

[`NeuralPredictor`](../../fusioneq/model.py) predicts 192 PCA coefficients of psi and is trained on a coefficient MSE.
R2psi is already 0.9998 (13th of 77), so the remaining psi error is small, but the scorer's seven consistency scalars
([`derive.py`](../../vendor/starter/fusion_scoring/derive.py)) are read off the LCFS contour and |grad psi|, which
weight that error very unevenly: an error near the boundary moves kappa, triangularity, volume and D_LCFS, and a
gradient error inside the plasma moves li, while the same error elsewhere costs nothing.

## Linearised consistency loss

psi is linear in the coefficients, psi = psi_vac + mean + (c_std * y) @ B, and at R2psi 0.9998 each scalar f_j is
close to linear in the coefficient error d around the label: f_j(pred) - f_j(label) ~ g_j . d with
g_j = d f_j / d y evaluated at the label. The consistency term's R2 per scalar is 1 - SSE_j / SST_j, so

    L_cons = sum_j  (g_j . d)^2 / var_j          (var_j: variance of f_j over training frames)

is, to first order, the consistency loss itself, and it is added to the coefficient loss with a weight w.

g_j is precomputed per training frame by central finite differences through the scorer's own `derive_frame`
(14 ms per call on the A100 host), stepping each coefficient by the model's typical error on it. This is exact by
construction (no reimplemented contour or O-point logic) and takes the secant at the error's own scale.

## Steps and exit tests

| | work | exit test |
|---|---|---|
| C1 | linearisation: on 200 held-out predictions of `v5_loss_mix_i03`, does g_j . d predict f_j(pred) - f_j(label)? | R2 > 0.8 per scalar. **Passes for the integral scalars only**: volume 0.90 and li 0.79 using the top 16 coefficients (97.7% of the error energy); kappa 0.21, Z_axis 0.37, R_axis / tri_top / tri_bot about 0 or below. The axis, elongation and triangularity are read off single extreme points (the O-point cell, the contour's highest / lowest / outermost points), so they jump rather than respond smoothly to a small psi error; volume and li integrate over the plasma. More coefficients only add finite-difference noise (all 192: volume 0.87, li -5.0) |
| C2 | precompute g for li and volume over the top 16 coefficients on a training subset (33 scorer calls per frame, ~110 k frames per hour on 15 cores) | cached |
| C3 | fine-tune `v5_loss_mix_i03` with L_cons on li and volume (gradients for 39,997 training frames), 10 epochs at lr 1e-4, weights 0 / 0.1 / 1 / 10 | li and consistency up with S not below 0.9915. **Failed.** The weight-0 control matches the base (S 0.9915, consistency 0.9673, li 0.965), so the fine-tune itself is harmless; every non-zero weight lowers held-out S (0.9905 / 0.9874 / 0.9828), R2psi and every consistency term, li included (0.956 / 0.946 / 0.952) while volume barely moves (0.970 / 0.967 / 0.966). The network cancels the li error on its training frames with psi changes that do not carry to unseen shots: the linearisation holds on held-out predictions (C1) but as a training signal it overfits |
| C4 | train fresh members with L_cons and ensemble them | not run: C3 failed |

## Risks

- The linearisation fails where the contour topology changes (X-point switching, limited vs diverted); C1 measures it.
- li's contour and region move with the prediction; holding them at the label's is part of the linearisation.
- Only li and volume linearise, so the reachable gain is about li 0.94 -> 0.96, i.e. S +0.0005 to +0.001.
