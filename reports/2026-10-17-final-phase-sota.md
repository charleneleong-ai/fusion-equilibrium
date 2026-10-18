# Fusion Equilibrium Challenge: Final-phase delta sweep, 2026-10-17

This is the delta on the [2026-10-11 sweep](2026-10-11-fusion-equilibrium-sota.md), written at 22:55 AEDT on 17 Oct.
Development closes at 11:00 AEDT on 18 Oct, and the Final runs from 11:00 AEDT on 19 Oct to 11:00 AEDT on 26 Oct.

Raw notes are in [`research_notes/2026-10-17-final-phase-sota/`](../research_notes/2026-10-17-final-phase-sota/):

- [`competition.md`](../research_notes/2026-10-17-final-phase-sota/competition.md): rules, issues, Final logistics.
- [`leaderboard/README.md`](../research_notes/2026-10-17-final-phase-sota/leaderboard/README.md): per-scalar
  breakdowns for all 146 teams, plus the published code.
- [`literature.md`](../research_notes/2026-10-17-final-phase-sota/literature.md): competitor repos, papers, solvers
  and priors.

Every number is as reported by its author or the organisers unless it is marked as ours. None of them was reproduced.

## 1. What changed, and the key findings

**The useful new material is competitors' published code and the public per-scalar leaderboard, not papers.**

1. **Two top-15 DIII-D teams published full pipelines, and both finish the map at the end.** These are Anulum
   (0.996359, [AGPL repo](https://github.com/anulum/fusion-equilibrium-challenge)) and Hakami (0.996223,
   [MIT repo](https://github.com/MuhammadHakami/fusion-equilibrium)).
   - **The finishing step.** Scalar heads are trained on the scorer's own derivation of the true maps. The ensemble ψ
     is then deformed so that the scorer reads the heads' values: the axis is shifted onto the predicted axis, the
     boundary is warped toward 64 predicted boundary rays, there are κ, δ and li warps, and the X-points are placed.
   - **We do none of this.** Our Consistency gap to the top (0.9813 vs 0.9897) is 0.0017 of our 0.0024 S gap, and li
     and κ alone are 0.0010 of it.
2. **There is a public, clean MAST method that scores 0.07 S above ours.** Anulum's "equal-Te" free-boundary solve
   reaches MAST S 0.7145 (R²ψ 0.979, Consistency 0.227), and their clean build scored 0.719.
   - The method: vacuum flux from coil geometry, plus plasma-current amplitudes fitted to Ip and to equal ψ at the
     inboard and outboard points of equal Thomson Te.
   - That isoflux constraint is the ingredient our closed free-boundary attempt lacked.
     [`mast-diagnosis.md`](../docs/mast-diagnosis.md#L90-L96) closed GS because "nothing holds the plasma" in place.
3. **Detailed Results are public for every leaderboard row**, so per-scalar breakdowns for all teams are now known.
   - The MAST cluster at S 0.80-0.81 (aidt_ipp, kim, argonclouds, yangyiliu) has R²ψ about 0.98-0.99, q95 R² 0.70-0.82,
     βN R² 0.33-0.50 and Consistency 0.45-0.50. Their methods are undisclosed.
   - Rows with MAST Consistency ≥0.73 (yszhang, dmitryzhevn, catskills, dremovd) remain implausible label-free.
4. **There is a Final-data quirk that our loader does not handle** ([issue #12](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/12)).
   - Two Thomson arrays in the Final inputs have non-monotone time: one DIII-D edge array is padded with a constant
     non-zero time, and one MAST core array has an out-of-order sample.
   - The organisers' supported fix is a strict running-max mask applied before any interpolation.
   - Our [`profile_on_efit`](../fusioneq/data.py#L74-L91) runs `searchsorted` on the raw times.
5. **Rules: nothing tightened.**
   - New since 10-11: a Final submission lost to a platform fault gets its slot refunded by email
     ([#11](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/11)).
   - A plagiarism clause now covers submitting another team's predictions.
   - The MAST solenoid "2×" factor is still an open caveat on the Data page.
   - The Final inputs are a new repo, `Sophelio/fusion-equilibrium-challenge-final`, published at 00:00 UTC on 19 Oct.
6. **The forum was not read.** It needs a login, and no account was used (see competition.md). Its two announcements
   are known only through the organisers' quotes in issues #8-#10.

## 2. Competition channels

The details are in [`competition.md`](../research_notes/2026-10-17-final-phase-sota/competition.md). Summary:

- **Issues.** There are 8, all closed, and the starter has no Discussions.
  - #5 and #6 are data errata, already applied through `data_fixes.py`.
  - #7 is the Ch2 gate.
  - #8, #9 and #10 cover rule 3 and rule 6. Selecting by Development score is allowed but must be disclosed, CPU
    synthetic data is uncapped, and the top 5 are rebuilt and checked against their provenance.
  - #11 is the platform-fault refund. #12 is the Thomson validity contract.
- **Data.**
  - v1.1.0 at revision `1e280905` is final; there will be no v1.1.1.
  - DIII-D Thomson uses 0 to mean "no measurement" (47% of edge rows and 23% of core rows are entirely zero). MAST uses
    NaN instead.
- **Scoring is unchanged.**
  - The ψ sign is normalised once per machine; the amplitude is not.
  - li uses the ITER li(2) form, κ and δ come from the LCFS bounding box, and the axis is the interior O-point.
  - A frame where derivation fails is mean-substituted.
  - float16 storage is safe. Fixed-decimal rounding is not.
- **Final logistics.**
  - 3 submissions, with the leaderboard hidden.
  - The Code URL must point to a public repo at a 40-character SHA, or a public archive with its SHA256.
  - An Award #2 nomination email is needed if the Ch2 pick is not the latest scored submission.

## 3. Leaderboard intel (146 teams, 11:45 UTC 17 Oct)

We are ranked 35th on Ch1 (0.994565). The top is 0.996998 (xiaoxiao5503) and 0.996946 (argonclouds); rank 20 is
about 0.9956.

DIII-D, our submission against #1 (weight × gap):

| li | κ | V | δ top/bot | R_axis | Z_axis | q95/βN | D_LCFS |
|---|---|---|---|---|---|---|---|
| .954 vs .975: 0.00059 | .970 vs .985: 0.00042 | .987 vs .995: 0.00022 | 0.00020 | 0.00013 | 0.00011 | .996 vs .9987: 0.00041 | .0097 vs .0073: 0.00024 |

MAST, our submission against the median of the 43 rows with 0.6 < S < 0.87:

| term | ours | median | weight × gap |
|---|---|---|---|
| R²ψ | 0.917 | 0.980 | +0.035 |
| Consistency | 0.127 | 0.272 | +0.029 |
| q95/βN | 0.219 | 0.414 | +0.029 |
| D_LCFS | 0.222 | 0.187 | +0.004 |

Against the 0.80-0.81 cluster, each of those gaps roughly doubles.

Published code and methods ([leaderboard README](../research_notes/2026-10-17-final-phase-sota/leaderboard/README.md)):

- **Full pipelines:** Anulum (both machines), Hakami (DIII-D only), and pshishkin's
  [MAST notes](https://github.com/pshishkin/fusion-equilibrium-challenge-starter/blob/main/experiments_mast.md). The
  MAST notes rely on demo labels, so only the ideas can be reused.
- **Method-name hints only:**
  - s_boehler (MAST 0.758): "kinetic GS, vertical class from P6 radial-field magnitude + temporal smoothing".
  - andregale (MAST 0.782): "Thomson-constrained Grad-Shafranov".
  - komakrv: "zero-shot GS, sol 0.5".
- **Dead links:** catskills' repo returns 404, and andregale's GitHub account has no public repos.

## 4. Literature (2025-2026 delta)

The papers add little to what we have; the details are in `literature.md`.

- **Architecture.** Shi et al. [2608.23217](https://arxiv.org/abs/2608.23217) find that CNN and FNO surrogates are
  the most stable out of distribution (4-5% relative L2), which fits our FNO members.
- **Physics constraints.** EFIT-Prime ([Phys. Plasmas 31, 092505](https://impact.ornl.gov/en/publications/efit-prime-probabilistic-and-physics-constrained-reduced-order-ne/))
  constrains J_φ and ψ smoothness, but it uses magnetics.
- **Input-seeded synthetic libraries.** ADITYA-U ([2607.04865](https://arxiv.org/abs/2607.04865)) and MAST-U
  ([2509.01789](https://arxiv.org/abs/2509.01789)) both seed solver libraries from real discharge inputs. That
  precedent is allowed for MAST only when the seeds come from inputs, never from EFIT.
- **Solvers.**
  - FreeGS v0.8.2 has a `MAST()` machine. To converge on raw MAST coils it needs P4/P5/P6 isoflux control and
    `check_limited=False` (Anulum ledger 5.18-5.20).
  - FreeGSNKE v3.1.1 still ships no MAST machine.
  - TokaMaker v26.9 includes an NSTX-U isoflux example, which is a permitted non-MAST spherical-tokamak prior.
- **Still missing:** no paper predicts li from non-magnetic data, and none transfers from a conventional to a
  spherical tokamak without labels.
- **Anulum's li ceiling.** Their own analysis puts it near 0.96-0.97. The residual is low-rank and persistent within a
  shot, which they read as information that only the magnetics carry.

## 5. Actionable levers, ranked by expected gain per effort

Constraints:

- About 12 h of Development remain. That window is the **last chance for any MAST feedback**; after the close, MAST
  changes can only be judged by the DIII-D labelled replay and label-free checks.
- The Final allows 3 blind submissions, with a 24 A100-h rebuild of which 16.0-18.8 h are already committed.
- Post-processing that runs on CPU costs no GPU budget.

### Challenge 1 (DIII-D)

| # | lever | evidence (author-reported) | expected gain | cost / risk |
|---|---|---|---|---|
| 1 | **Finish ψ toward scalar heads:** axis shift (Gaussian-bump displacement of ψ around the O-point, σ 0.15 m, capped at 2 cells, iterated against the scorer's own `find_o_point`), then a joint Gauss-Newton warp on (R_axis, Z_axis, κ, δ_top, δ_bot, V) normalised by label spread, then a monotone interior li remap that leaves the LCFS and axis fixed. Accept per frame only if the warped map's derived scalars move toward the heads. Heads: fit ridge or GBDT on our blend's features or last activations, with labels = [`scorer_scalars`](../fusioneq/consistency.py) of the true maps | Hakami [`axis_shift.py`](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/paper/strat/fusion/axis_shift.py), [`joint_warp.py`](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/paper/strat/fusion/joint_warp.py): sequential warps realise Consistency 0.98419 of the heads' 0.98711, and the joint fit removes the cross-talk. Anulum's finishing step: [METHODS_REPORT §2](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/docs/METHODS_REPORT.md) | Consistency 0.981 → 0.984-0.987: **+0.0006-0.0012 S** (heads must beat the map's own derived scalars; ours already sit at li .954 / κ .970) | 1-2 days CPU; no GPU; no retrain. The gate must be the 160 held-out shots, because no public feedback exists after 11:00 tomorrow. Screen-to-official transfer has been about 0.5× |
| 2 | **Dedicated q95/βN ensemble + clamp.** Huber (δ 0.3) MLP heads; blend about 0.5 into q95 and 0.9 into βN; clamp q95 and βN to the training-label range | Anulum: 8 dedicated nets, shares 0.46 / 0.87. Hakami: `qb_clamp.json`. Top q95 / βN R² is .9988 / .9986 against our .9960 / .9959 | **+0.0002-0.0004 S** | Small MLPs, under 1 A100-h; the clamp is free |
| 3 | **li GBDT blended into the li head** (w 0.2-0.26) and κ GBDT (w 0.09) | Anulum +3.5e-5 S; Hakami w 0.2 | +0.00003-0.0001, and only together with #1 | CPU |
| 4 | **Ip-sign canonicalisation:** multiply Ip, F-coils and E-coil (not the B-coil) by sign(mean Ip) per shot; targets unchanged | Hakami [`canon.py`](https://github.com/MuhammadHakami/fusion-equilibrium/blob/main/paper/strat/fusion/canon.py): 8.7% of public-test shots are reversed; routing them to canonical heads gave 0.995949 → 0.996167 (+2.2e-4) | +0.0001-0.0002 | Retrains the MLP members (GPU budget). Feasible only if #1 leaves room; otherwise apply it to the head models only |
| 5 | **Sub-frame features:** 0.5 ms signals within ±20 ms of each frame | Anulum ledger 2.41: +2.0e-4 S, D_LCFS .00889 → .00848 | +0.0001-0.0002 | Retrain plus feature rebuild; same GPU-budget caveat |
| 6 | **Thomson monotone-time mask** (organiser snippet, issue #12) | Two arrays in the Final inputs are non-monotone | Prevents a silent misread on 1 DIII-D and 1 MAST shot | Minutes. To stay byte-consistent with training, apply it at inference and confirm that nothing changes on the public fold except the 10 training rows |

### Challenge 2 (MAST, label-free)

| # | lever | evidence | expected gain | cost / risk |
|---|---|---|---|---|
| 1 | **Equal-Te isoflux constraints.** Fit per frame the plasma-current parameters (centroid R, width, peaking, Ip-normalised) in our Green's-function model so that ψ is equal at the inboard and outboard midplane Thomson points of equal Te, at several Te levels (0.85 down to 0.05 of peak), with Te at 5% of peak as the LCFS marker. Use a linear (1−ψN) profile. Keep currents vanishing at the boundary | Anulum v7/v11 ([ledger §5](https://github.com/anulum/fusion-equilibrium-challenge/blob/main/docs/EXPERIMENT_LEDGER.md)): S 0.714-0.730, R²ψ 0.979-0.981, Consistency 0.23-0.27, 88-90% of frames solved; Te 5% sits −1.7 cm from the LCFS on DIII-D truth; (1−ψN)² dropped S to 0.601; edge-current terms broke Picard. The same constraints also set the flux span, which pshishkin found drives 5 of 7 scalars' sign errors | R²ψ 0.917 → ~0.97, Consistency 0.13 → ~0.2: **+0.04-0.06 S_MAST ≈ +0.04-0.06 G** | 1-3 days CPU. One Development submission before 11:00 tomorrow is the only official check. Otherwise gate it on the DIII-D labelled replay (`shape-check`), which predicted 688a773d's official gain. Reimplement from the published description: the code is AGPL and the plagiarism clause covers predictions |
| 2 | **βN = 2.25 × Thomson p_e** volume-averaged on the solved map's flux surfaces (factor fitted on DIII-D); **q95 read from the solved map**, no Uckan formula (it diverges as a/R0 → 1) | Anulum v10/v11 q95/βN 0.32-0.40 vs our 0.22; the MAST cluster reaches q95 .70-.82 | +0.015-0.03 S | CPU; depends on #1 |
| 3 | **Vertical placement from the P6 radial-field coil current** (sign/magnitude → up/down class), plus temporal smoothing of placement | s_boehler's method name (MAST 0.758). Z_axis is ≤0 for most of the band | Z_axis R² 0 → 0.2-0.4: +0.006-0.01 S | Cheap; geometry and inputs only |
| 4 | **Solenoid at half the released current in the shape term** (already in 4c428222's shape part). Keep the level as it is | Data-page 2× caveat; Anulum and komakrv both use 0.5 | Already harvested | — |

## 6. What does NOT apply

- **Coil and vessel freedom fitted to the isoflux pairs, and other gains that only appear on synthetic FreeGS
  testbeds.** Anulum saw −38-45% D on synthetic data but lost on the real board (0.7246 vs 0.7296). Our own
  vessel-circuit result was also closed.
- **(1−ψN)² or edge-current profile terms**, which reduce the solved share and lower S.
- **Uckan/ITER q95 from an approximate MAST boundary**, which diverges.
- **Temporal smoothing of DIII-D scalars.** Anulum found every scalar worse. MAST placement smoothing is a separate
  matter.
- **More physics features for DIII-D li and κ** (βp proxy, Te^1.5 history, current diffusion, attention over the whole
  shot). Anulum found no gain. Hakami's Spitzer Te^1.5 features are untested in isolation.
- **pshishkin's span and γ fits, MAST-U βN ranges, and the MAST kinetic-EFIT database.** All rest on MAST or MAST-U
  EFIT labels (demo shots or archive) and are barred by rule 3. Only the mechanism ideas are reusable.
- **Copying Anulum's code or weights.** The AGPL licence and the "own team's work" clause apply; reimplement the ideas
  and disclose them.
- **TokaMind / TokaMark weights**, which remain a rule-3 risk.

**Does this change the plan? Yes.**

- Before the Dev close: if one slot remains, spend it on MAST lever 1 in its cheapest form (an isoflux fit of the
  existing ellipse parameters). It is the only remaining official MAST signal.
- During the Final week: build DIII-D lever 1 (CPU finishing step, gated on the 160 held-out shots) and the Thomson
  mask. Neither touches the 24 A100-h budget.
- **Next sweep:** none needed before the close. If one is done, read the forum through a logged-in browser for
  Final-phase announcements on 19 Oct.
