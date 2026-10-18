# Literature and competitor delta sweep, 2026-10-17

Delta on [2026-10-11 notes](../2026-10-11-fusion-equilibrium-sota/literature.md). All numbers are author-reported
(arXiv abstract, repo README or ledger) and were not reproduced. "abstract only" means the full text was not read.
Status column: NEW = not in the 10-11 sweep; KNOWN = already covered.

## 1. Competitor repositories (the most actionable part of this sweep)

| repo | status | machine | headline (author-reported, Development public fold) | concrete tricks |
|---|---|---|---|---|
| [anulum/fusion-equilibrium-challenge](https://github.com/anulum/fusion-equilibrium-challenge) (created 2026-10-14; weights [HF anulum/fusion-equilibrium-challenge](https://huggingface.co/anulum/fusion-equilibrium-challenge)) | NEW | both | DIII-D S 0.996359 (R²ψ .999883, qb .998005, D_LCFS .008180, Cons .987704), 10th/129 on 10-14. **MAST S 0.714450** (R²ψ .97945, qb .3307, D .1918, Cons .2266), best MAST ever logged 0.7296 (v7, pre-rule choices) | see §1a |
| [MuhammadHakami/fusion-equilibrium](https://github.com/MuhammadHakami/fusion-equilibrium) (pushed 2026-10-17) | NEW | DIII-D only | F15K 0.995949 → **F16K 0.996167 (+2.2e-4) from Ip-sign routing**; F17K adopted for Final | see §1b |
| [pshishkin/fusion-equilibrium-challenge-starter](https://github.com/pshishkin/fusion-equilibrium-challenge-starter) | NEW | both | MAST S ≈ 0.568 (GS solve), Consistency 0.0 on every entry; [experiments_mast.md](https://github.com/pshishkin/fusion-equilibrium-challenge-starter/blob/main/experiments_mast.md) | see §1c |
| [rongtnt/fusion-equilibrium-2026](https://github.com/rongtnt/fusion-equilibrium-2026) | NEW | DIII-D | PCA-50 + Ridge + ExtraTrees residual, local S ≈ 0.949 | none new |
| [SII-AI4Fusion fork](https://github.com/SII-AI4Fusion/fusion-equilibrium-challenge-starter) (+14 commits) | NEW | DIII-D | full-corpus PcaScalarMLP baseline | none new |
| [Dibyajit7 fork](https://github.com/Dibyajit7/fusion-equilibrium-challenge) (+12 commits) | NEW | both | S 0.9576 DIII-D; MAST "machine-agnostic physics features", SSIM-only, n=3 demo shots | weak |
| [HaoPhuc](https://github.com/HaoPhuc/fusion-equilibrium-challenge), [RimmaShaf fork](https://github.com/RimmaShaf/fusion-equilibrium-challenge-starter) (+2, EDA notebook) | NEW | — | plan / EDA only | — |
| other 7 forks of the starter | — | — | 0 commits ahead of upstream (checked with the compare API) | — |

Starter upstream last commit [`ab4e2c5`](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/commit/ab4e2c5) (2026-10-11): "v1.1.0 + data_fixes.py is final".

### 1a. Anulum ([METHODS_REPORT](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/docs/METHODS_REPORT.md), [EXPERIMENT_LEDGER](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/docs/EXPERIMENT_LEDGER.md))

DIII-D:
- 582 features per frame: currents + rates, currents/Ip, discharge phase, Thomson on fixed bins, windowed means
  (±0.08 s, 0.3 s, 1.3 s) and **215 sub-frame features from the 0.5 ms signals within ±20 ms**. The sub-frame block was
  the biggest late gain: S .996025 → .996223 (+2.0e-4) and D_LCFS .00889 → .00848 (ledger 2.41).
- 69 MLPs (512×4) on 64 PCA coefficients, keep the leading 45. Auxiliary heads for axis, κ, δ, volume, li, X-points,
  q95/βN, 64 boundary rays; heads refitted by ridge on last activations. **Shape labels are the scorer's own
  derivation applied to the true maps.**
- GBDT heads blended at fixed weights: li .26, R_axis .14, κ .09, lower-X R .16 (+3.5e-5 S, Cons .98652 → .98668).
- q95/βN: dedicated 8-net ensemble blended at .46 (q95) / .87 (βN).
- Finisher edits the mean map so the scorer reads predicted axis, boundary, shape, li and X-points.
- **Remaining limit: li .963, κ .980, R_axis .983 (OOF R²)**. Things that gave nothing for li/κ: 27 physics features
  (βp proxy, pedestal, Ip and Te^1.5 histories, 1-D current diffusion), temporal smoothing of scalars (li .9717 →
  .9677 at w=3), 4× scalar label weight (li +1.2e-3 but δ and D_LCFS worse, net 0), electron-pressure chords,
  attention over all frames of a shot, exponentially weighted history 0.3-10 s.
- The residual map after their model is low-rank (3 directions = 83 % of energy), persistent within a shot, and
  not recovered from any input; their reading is that it is magnetics-only information.

MAST (zero-shot, "equal-Te" solve, v7 to v12):
- Vacuum flux from the released coil geometry + a plasma current whose amplitudes are fitted to measured Ip and to
  **equal ψ wherever the midplane Thomson profile reads equal Te** (inboard/outboard Te pairs = flux-surface isoflux
  constraints). q95 read from the map; **βN = Thomson p_e × 2.25** (total/electron factor set on DIII-D).
- **Solenoid share per conductor row 0.5**, re-derived without labels from the organisers' statement that the
  released solenoid current reads twice the archive value.
- Te 5 % of peak marks the LCFS on DIII-D truth at −1.7 cm (IQR 1.6 cm); on MAST the Te-5 % radii sit inside the
  LCFS (≈0.10 m inboard, ψN ≈ 0.91).
- Profile basis matters: switching to (1−ψN)² dropped MAST S .719 → .601 and the solved share 90 % → 68.5 %; adding
  non-vanishing edge p'/FF' terms destabilised Picard (solved share .85 → .22).
- Coil/vessel freedom: fitting P4/P5 (±3 %) and a vessel current (∝ Ip) to the equal-Te pairs gave D −38..−45 % on
  a FreeGS synthetic MAST testbed and passed label-free guards on real inputs, but **on the public board v10 was
  S .7246 vs v7 .7296**: the synthetic gain did not transfer.
- βN from the map's GS split (p' vs FF') was far worse than kinetic (R² .43 vs .90 on DIII-D truth).
- FreeGS driven by raw MAST coils: 6 of 19 frames put the plasma at the outer field null; it needs shaping-coil
  (P4/P5/P6) isoflux control plus `check_limited=False` (diverted) to converge 18/19 frames.
- Uckan/ITER q95 formula on an approximate boundary diverges as a/R0 → 1 (withdrawn).

### 1b. Hakami F17K ([repro README](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/repro/fusion/README.md))

- **Ip-sign canonicalisation** ([`canon.py`](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/paper/strat/fusion/canon.py)):
  multiply Ip, the 18 F-coils and ECOILA (not the B-coil) by sign(mean Ip) per shot. This is an exact GS symmetry.
  The dataset ψ is orientation-normalised and all scored scalars are invariant. 8.6 % of training and 8.7 % of
  public-test shots have reversed Ip. Routing those shots to "canonical twin" heads: +2.2e-4 S.
- **Huber loss (δ 0.3) on shape heads and q/βN heads.**
- **li features from Spitzer scaling** ([`liphys_feats.py`](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/paper/strat/fusion/liphys_feats.py)):
  ohmic j ∝ σE with σ ∝ Te^1.5, so they take Te^1.5 and p centroid/RMS width on the vertical core chord, peaking
  factors, and log core Te^1.5 (13 columns).
- CatBoost li tree blended 0.2 into head li; a second κ/li warp pass; q95/βN clamped to the label range.
- Post-processing: axis shift, boundary warp, X-point warp, κ warp, Gauss-Newton joint shape fit, li warp ×2.

### 1c. pshishkin MAST notes

- **The flux span (ψ_axis − ψ_bnd) is the root of the Consistency failure.** On their solve, 5 of 7 scalars have the
  wrong sign of correlation and only li moves the right way. The truth span grows 2.4× at flat Ip while theirs stays
  constant, because the profile shape is fixed and normalised to Ip. Lever: let the profile exponent vary per frame.
  Their sweep moves the span 5.5×.
- MAST first wall from FAIR-MAST geometry: +0.0025 S. Geometry is permitted, but that archive also holds the
  withheld truth (`efm/psirz` is bit-identical).
- An `edge = 0.03` source extending 3 % of the span past the LCFS: +0.0083 S.
- q95 contour bug (R² instead of R): watch for it. Fitted B0 = 0.366 T at an 85 kA TF feed. Anulum measures a
  median vacuum B0 of 0.479 T at 0.85 m.
- **Caution:** their scalar affines (q95 ~ exp(c + w·log[|I_TF| q_shape, V, Ip]), βN ~ V, 1/Ip, a) were fitted on the
  3 MAST demo shots. The organisers now say demo use is "format inspection only" (issue #8), so this is not reusable.
  The structural finding that volume correlates −0.90 with βN within each shot is only a hypothesis to test on
  synthetic data.

### 1d. Rule clarifications in starter issues (primary: organiser replies)

- [#8](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/8) / [#9](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/9), forum [2807](https://www.codabench.org/forums/17429/2807/):
  MAST/MAST-U reconstructions from **any** source (FAIR-MAST, mirrors, derived weights) may not train, calibrate,
  validate or select. The demo shots are covered too. **Allowed with disclosure:** MAST inputs for self-supervised
  learning, target normalisation, distribution alignment, **label-free TTA such as physics-residual optimisation**,
  and synthetic MAST equilibria whose generator ranges are calibrated against MAST *diagnostic inputs* only. Separate
  DIII-D and MAST pipelines are fine. `G_ratio` is uncapped in code (≤ 1/0.85).
- [#10](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/10), forum [2827](https://www.codabench.org/forums/17429/2827/):
  CPU synthetic-data generation (e.g. FreeGSNKE, 600 core-h) is not counted in the 24 GPU-h limit. Ship it with
  SHA256, seeds and the solver version. Another team is training a MAST model on FreeGSNKE synthetic data.
- [#7](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7): the G_ratio gate is on the full
  DIII-D S ≥ 0.85.

## 2. Papers

| work | status | setting | result (author-reported) | relevance |
|---|---|---|---|---|
| Shi et al., [arXiv 2608.23217](https://arxiv.org/abs/2608.23217) (2026-08-24) | NEW | EXL-50U; MLP/CNN/FNO/Transformer/KAN benchmark on 100k IID + 10k OOD GS equilibria | surrogate error 1e-3–1e-2; **OOD: CNN and FNO most stable at 4-5 % rel. L2**; Transformer best IID; scaling helps interpolation, not extrapolation. Abstract only | architecture choice for synthetic-MAST surrogates: prefer CNN/FNO for OOD |
| Maurya et al., [arXiv 2607.04865](https://arxiv.org/abs/2607.04865) (2026-07-06) | NEW | ADITYA-U, 100,760 synthetic pyIPREQ equilibria from 766 discharges; DNN/PCA-ROM/CNN/PINN with GS residual | no numbers in abstract | precedent: train on solver equilibria seeded by real discharge inputs |
| Kaltsas et al., [arXiv 2609.23846](https://arxiv.org/abs/2609.23846) (2026-09-20) | NEW | PI-KAN, fixed-boundary GS, homotopy curriculum, profile-parameter identification | no numbers in abstract | low |
| Abate, [arXiv 2609.15074](https://arxiv.org/abs/2609.15074) (2026-09-14) | NEW | analytic Solov'ev, non-convex boundaries | Shafranov-shift sign reversal for odd-fold concave shapes | low (analytic) |
| Madireddy, Akçay, Lao et al., EFIT-Prime, [Phys. Plasmas 31, 092505 (2024)](https://impact.ornl.gov/en/publications/efit-prime-probabilistic-and-physics-constrained-reduced-order-ne/) | NEW (missed) | DIII-D, magnetics, NAS deep ensemble; **J_φ constraint + smooth ∂ψ** physics loss; diagnostic locations as inputs | generalises to held-out negative-δ shots; 15 ms/ensemble | ensemble + GS-current regulariser precedent; magnetics-based |
| Lu et al., EAST, [arXiv 2305.12098](https://arxiv.org/abs/2305.12098) | NEW (older) | EAST magnetics → EFIT ψ; li computed from the predicted ψ and compared with EFIT | — | li-from-ψ consistency check precedent |
| [arXiv 2503.05674](https://arxiv.org/abs/2503.05674) | NEW | multiple solutions of the static forward free-boundary GS problem on MAST-U (FreeGSNKE) | abstract only | explains Picard non-uniqueness and failure modes when synthesising ST equilibria |
| [arXiv 2509.01789](https://arxiv.org/abs/2509.01789) | NEW | MAST-U emulated virtual circuits; FreeGSNKE libraries **seeded by MCMC from real MAST-U snapshots**, ~1.8e6 equilibria | abstract only | recipe for input-seeded synthetic libraries; seeding from MAST *inputs* is allowed, from EFIT is not |
| Ross et al. [2605.14939](https://arxiv.org/abs/2605.14939), Stokolesov [2505.10709](https://arxiv.org/abs/2505.10709), organisers [2609.01750](https://arxiv.org/abs/2609.01750), liuqe-nn [2606.09487](https://arxiv.org/abs/2606.09487), Krastev [2608.05555](https://arxiv.org/abs/2608.05555), Yoo [2606.15512](https://arxiv.org/abs/2606.15512), Wang [2608.23976](https://arxiv.org/abs/2608.23976), TokaMark/TokaMind | KNOWN | — | — | no new versions found |

Not found (gap): no 2025-26 paper that predicts li from non-magnetic data with ML, or that transfers
conventional → spherical without target labels. TokaMark sensor-robustness follow-up
[2607.11915](https://arxiv.org/abs/2607.11915) is NEW but about MAST diagnostics robustness, not equilibria.

## 3. Spherical-tokamak priors (c)

- NSTX ([Gerhardt et al. NF 2011/2012](https://nstx.pppl.gov/DragNDrop/Publications_Presentations/Publications/2011%20Papers/2011_locked/Gerhardt_NF_1.pdf)):
  κ ~2.3 in earlier campaigns, ~2.6-2.7 high-κ targets after lithium (with lower li); shape factor
  S = q95·Ip/(a·Bt) of 34-40 (record ~41 at κ ~3, [FEC 2006](https://www-pub.iaea.org/MTCD/Meetings/FEC2006/ex_p1-3.pdf)).
  Non-MAST machine, so permitted as a prior.
- MAST-U ([Berkery ISTW 2022](https://www.afs.enea.it/project/protosphera/Proto-Sphera_Full_Documents/STW2022-Cina/Presentazioni/Berkery-MASTU_ISTW2022_v1.pdf)):
  βN up to 3.5, βN/li up to 3.4. **Gray zone:** these are MAST-U numbers derived from EFIT. Using a published range
  is not training on reconstructions, but it is MAST-family label information. Only use it as a loose sanity bound,
  disclose it, and never use it to calibrate.
- The MAST kinetic-EFIT database (Berkery PPCF 2021, 275 equilibria) is MAST EFIT. **Do not use it.**
- Anulum's synthetic testbed used a li prior of 0.8-1.3.

## 4. Free-boundary solvers (d)

| solver | latest | MAST machine? | note |
|---|---|---|---|
| [FreeGS](https://github.com/freegs-plasma/freegs) | v0.8.2 (2024-09-25); main pushed 2026-10-05 | `MAST()`, `MAST_sym()` in `machine.py` | Anulum: raw MAST coils need P4/P5/P6 isoflux control and `check_limited=False` to converge |
| [FreeGSNKE](https://github.com/FusionComputingLab/freegsnke) | v3.1.1 (2026-09-24); main pushed 2026-10-16 | MAST-U-like, ITER, SPARC configs only; no MAST | another team uses it for MAST synthetic data (issue #10) |
| [OpenFUSIONToolkit / TokaMaker](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) | **v26.9 (2026-09-08)**, v26.6, v1.0.0-beta7 | **NSTX-U geometry, mesh and isoflux-controller example** (`src/examples/TokaMaker/AdvancedWorkflows/IsofluxController/NSTXU_*`); DIII-D, LTX and HBT examples; no MAST | NSTX-U is a non-MAST ST, so its equilibria are permitted as transfer data |

## 5. Hugging Face

Models: [anulum/fusion-equilibrium-challenge](https://huggingface.co/anulum/fusion-equilibrium-challenge) (NEW,
AGPL weights) and our own `chaleong/fusion-equilibrium`. No new datasets apart from the challenge set.
