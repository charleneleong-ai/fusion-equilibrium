# Fusion Equilibrium Challenge: SOTA sweep, 2026-10-11

First SOTA report for the entry (Codabench [17739](https://www.codabench.org/competitions/17739/), organiser Sophelio).
Sources and raw tables are in [`research_notes/2026-10-11-fusion-equilibrium-sota/`](../research_notes/2026-10-11-fusion-equilibrium-sota/):
[`competition.md`](../research_notes/2026-10-11-fusion-equilibrium-sota/competition.md) (rules, organiser answers,
leaderboard snapshot) and [`literature.md`](../research_notes/2026-10-11-fusion-equilibrium-sota/literature.md)
(papers, code). Primary sources only, and every number is author- or organiser-reported unless marked as ours.

**Bottom line.** No published method reconstructs equilibria without magnetics at anything near leaderboard
accuracy. The only DIII-D reduced-diagnostic paper reports 3-4 cm *mean* LCFS error, against our 1.9 cm Hausdorff
*max*. Nothing published transfers DIII-D to MAST without target labels. The useful findings are about the
competition itself:

1. On MAST, the rule-3-plausible peers sit at G 0.70-0.86, and nearly all of their lead over us is **Consistency**
   (median 0.22 vs our 0.012) and **R²ψ** (median 0.974 vs our 0.908).
2. The rules tightened on 4 Oct. Two pieces of our MAST path, settings chosen on the demo shots and demo-fitted
   calibrations, must be re-justified or removed before the Final.

## 1. Where we stand (official Development leaderboard, 2026-10-11 08:00 UTC)

| | S | R²ψ | R²{q95,βN} | D_LCFS | Consistency |
|---|---|---|---|---|---|
| DIII-D, us (rank 29 / 50) | 0.9939 | 0.9998 | 0.9959 | 0.0110 | 0.9788 |
| DIII-D, top 5 | 0.9964-0.9967 | 0.9999 | 0.9980-0.9984 | 0.0068-0.0077 | 0.9874-0.9881 |
| MAST, us (G 0.6316) | 0.6278 | 0.908 | 0.320 | 0.220 | 0.012 |
| MAST, median of 26 peers with 0.5 < S < 0.85 | ~0.73 | 0.974 | 0.320 | 0.199 | 0.223 |

How the DIII-D gap to the leader (0.0028) splits by term, weight × difference (ours):

| term | contribution |
|---|---|
| Consistency | 0.20 × 0.0093 = **0.0019** |
| D_LCFS | 0.10 × 0.0042 = 0.0004 |
| q95/βN | 0.15 × 0.0025 = 0.0004 |
| R²ψ | 0.0001 |

On MAST, matching the peer median on Consistency alone is worth **+0.042 S_MAST ≈ +0.042 G**, and on R²ψ
+0.036. That is about 15× the entire DIII-D gap.

Three MAST rows can't be taken at face value. jygao (MAST S = 1.0000) and dmitryzhevn / catskills (D_LCFS 0.04-0.05,
Consistency ~0.88) are beyond anything label-free we know of. Rule 3 is not applied retroactively to Development
entries, but the Final rebuild of the top five per challenge will apply it. The realistic Final competition is the
0.70-0.86 band.

## 2. Literature, 2024-2026 (details and links in `literature.md`)

**Magnetics-free / reduced diagnostics.**
- [Stokolesov et al. 2505.10709](https://arxiv.org/abs/2505.10709) (DIII-D, ~25k shots) is the only direct
  precedent: coil currents give a 0.04 m mean LCFS displacement, and adding Ip and V_loop gives 0.03 m.
- [Sun et al., NF 64 (2024)](https://doi.org/10.1088/1741-4326/ad5d7b) (EFIT-AI kinetic surrogate) finds extra
  diagnostics add little once magnetics are in; it does not test the no-magnetics case.
- **Takeaway:** we are past the published state of the art on DIII-D, so there is nothing to import directly.

**Surrogates and operator learning.** The 2025-26 trend is neural operators and branch/trunk physics-regularised
networks:
- FNO on free-boundary FreeGS equilibria: 0.05% relative L2, X-point within 0.2 cm, error ∝ N^-0.68
  ([2608.05555](https://arxiv.org/abs/2608.05555)).
- PINO / Transformer-KAN: 0.25% relative L2; physics loss cuts residuals ~10^4×
  ([2511.19114](https://arxiv.org/abs/2511.19114)).
- TCV liuqe surrogate, under 100 µs, with [code](https://github.com/matgrand/liuqe-nn)
  ([2606.09487](https://arxiv.org/abs/2606.09487)).

All of these use magnetics or synthetic data. Their relevance to us is architectural: a structurally different grid
family, which is the kind of change that moved our score twice before.

**Cross-machine.**
- [Yoo et al. 2606.15512](https://arxiv.org/abs/2606.15512): multi-geometry pretraining transfers and
  single-geometry pretraining "transfers poorly". It still needs ~100 target equilibria to reach under 4% error,
  which rule 3 forbids for MAST. That matches our closed result that a machine-agnostic grid net trained on one
  machine does not place the plasma.
- MAST-U emulators train on >10^6 synthetic FreeGSNKE equilibria
  ([2605.14939](https://arxiv.org/abs/2605.14939)). Another team does the same for this challenge
  ([issue #10](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/10)), but nobody has published
  a result.

**Avoid.** TokaMark/TokaMind ([2602.10132](https://arxiv.org/abs/2602.10132),
[2602.15084](https://arxiv.org/abs/2602.15084)) are pretrained on FAIR-MAST. If EFIT signals were in that pretraining,
the weights fall under rule 3's "model weights trained on" clause.

**Q95, βN, li, κ.** No 2024-26 paper estimates li or κ without magnetics. HL-3 with magnetics gets mean R² 0.941
over 8 scalars ([2405.11221](https://arxiv.org/abs/2405.11221)), well below what we already score on DIII-D.

**Code and checkpoints.**
- No equilibrium checkpoints on Hugging Face, and no competitor solution repos.
- Useful public geometry: FreeGS ships `MAST()` / `MAST_sym()` machine descriptions
  ([machine.py](https://github.com/freegs-plasma/freegs/blob/master/freegs/machine.py)), which is an independent check
  on the MAST coil and solenoid geometry the challenge ships.
- FreeGSNKE [v3.1.1](https://github.com/FusionComputingLab/freegsnke) (2026-09-24) ships MAST-U, not MAST.

## 3. Rules that changed under us (Codabench Rules page; issues #8-#10)

Rule 3 now reads (effective 09-23; feedback clause 10-01; probe and mask clauses 10-04):

- **The demo shots are for format inspection only.** No validation, selection or calibration on them.
- **Platform scores from the Development phase may be used** to compare, tune and select MAST pipelines, as long as
  the methods report says so.
- **Probes are banned.** Constant, shifted or rescaled submissions built to read label statistics, and any constant
  fitted to what they reveal, count as MAST equilibrium data.
- **`mast_envelope.npz` may only be used inside scoring.**
- **Label-free test-time adaptation is explicitly allowed**, e.g. physics-residual optimisation on MAST inputs
  ([#8](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/8)).
- **The DIII-D and MAST files may come from different pipelines**
  ([#9](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/9)).
- **Award #2 divides by the best DIII-D S across our Final submissions.**

Compliance actions for the Final package. These are preconditions, not levers:

1. **Re-justify alpha 2.0 / frac 0.4 / eps cap.** They were picked by a 36-setting sweep on the demo shots
   ([`mast-diagnosis.md`](../docs/mast-diagnosis.md)). Keep them only because official Development scores back them
   (`d32bdf0c` 0.6045 vs `501b7986` 0.6017; eps 0.65 vs 0.7 was also scored officially), and say so in the methods
   report.
2. **Prove two demo-fitted classes are off in the Final MAST path:**
   [`SolenoidOffset`](../fusioneq/physics.py#L89-L115) and [`DemoScalarFit`](../fusioneq/physics.py#L116-L130).
   A test asserting the Final config never loads `mast_offset.json` or a demo fit would settle it.
3. **Keep `mast_envelope.npz` out of anything that produces MAST predictions.** Today only
   [`fusioneq/score.py`](../fusioneq/score.py#L25) and the closed `agnostic.py` path read the scorer masks.
4. **Declare a rebuild budget of at most 24 A100-hours** for the full DIII-D stack: 27 MLP members, the trees and
   three grids.

## 4. What changed / does it change the plan

**What changed** (first sweep, so this is measured against our working assumptions):

- The literature offers no drop-in method. The SOTA for this task *is* the leaderboard.
- The rule-3-plausible MAST band (G 0.70-0.86) is set mainly by Consistency and R²ψ, not by q95/βN, where we already
  sit at the peer median of 0.32.
- Rule 3's demo, probe and mask clauses put two of our MAST settings at risk. See the preconditions above.
- Another team is building synthetic MAST equilibria with FreeGSNKE. It is permitted, and its CPU time is uncapped.

**Does it change the plan? Yes.** The work shifts toward MAST Consistency. The comparison for a week's effort:

- **MAST:** +0.04 G is plausible.
- **DIII-D:** the realistic ceiling is about +0.0005 S.

The DIII-D work stays because it is cheap and already validated. The ranked levers below have to fit in the 6
Development days left (5 submissions a day, closing 2026-10-18 00:00 UTC) and the Final's 3 submissions and 24
A100-hour rebuild budget.

| # | lever | term moved, expected gain | cost | rule 3 |
|---|---|---|---|---|
| 1 | **MAST Consistency: diagnose, then fix shape.** (a) Read the Detailed Results of `4c428222` for per-scalar R² and the **derivation-failure rate**. Failed frames are mean-substituted and earn about 0. If failures are non-trivial, guarantee one interior O-point and a closed LCFS with an X-point pair in our ψ. (b) Replace the fixed ellipse's κ, δ and Z centroid with a dimensionless map from vacuum-field features (decay index at the Thomson-placed centroid, up/down P-coil asymmetry), fitted on **DIII-D** labels and applied to MAST. | MAST Consistency 0.012 → 0.1-0.2: **+0.02-0.04 G**; small D_LCFS gain | (a) free; (b) 1-2 days CPU, 2-3 Development submissions | Compliant: DIII-D calibration, MAST inputs and geometry, selection by platform scores (disclose) |
| 2 | **Ship the grid re-weight (0.75)** plus whichever in-flight grid variant (heavier envelope weight, ±8 frames; the 96-channel run was dropped - it does not fit beside other jobs on the shared GPU) wins most 40-shot groups on the 160 held-out shots (gate G2) | DIII-D Consistency and D_LCFS, **+0.00007-0.0002 S** (0.75 alone: +0.000086 on 160 shots, every group) | ~0 extra GPU (in flight); 1-2 submissions | n/a (DIII-D) |
| 3 | **Stacked q95/βN on predicted geometry.** A GBDT on out-of-fold blend ψ → scorer `derive_frame` scalars (κ, δ, V, a, li, R_axis), plus B_T coil, Ip and a Thomson stored-energy proxy ∫nₑTₑ over the predicted V. This is not the closed "direct heads": the new input is self-consistent geometry, which the textbook q95 ∝ a²B_T(1+κ²)/(R Ip) and βN = β_T a B_T / Ip need. | DIII-D q95/βN 0.9959 → ~0.997-0.998: **+0.0002-0.0003 S** (q95/βN has been flat since 30 Sep) | CPU, ~0.5 day; G2 on 160 shots | DIII-D only; keep textbook q95/βN on MAST unless a Development submission says otherwise |
| 4 | **MAST R²ψ level from geometry, not labels.** Cross-check the shipped MAST solenoid and passive geometry against FreeGS `MAST()`. If it is short (turns, height), fix it from geometry. Optionally add a passive-structure circuit term driven by dI_sol/dt. Every constant must come from geometry or physics. | MAST R²ψ 0.908 → ≤0.95: **up to +0.02 G** (low confidence; the diagnosis says MAST ships its solenoid turn by turn) | 1 day CPU, 1-2 submissions | Compliant only if no constant is fitted to scores. A level sweep by submission is a **probe** |
| 5 | **Neural-operator grid member** (FNO/WNO head on the same shape target and FiLM inputs) as a fourth family, blended under G2 | DIII-D Consistency and D_LCFS, **+0.0001-0.0004 S** (speculative; earlier new families gave +0.0006 and +0.0008) | ~2-3 A100-h per seed × 3; must fit the 24 h rebuild; only if 2-3 leave room | n/a (DIII-D) |

**Not this week:**

- **Synthetic FreeGS/FreeGSNKE MAST training set.** Permitted, but our own free-boundary solve did not track EFIT's
  shape scalars even on DIII-D (closed), and a synthetic-to-real model cannot be validated under rule 3 in 6 days.
- **TokaMind / TokaMark weights.** Rule-3 risk.
- **PINN / PINO physics losses.** Validated only on analytic equilibria; the GS loss is closed for us.
- **Any MAST offset or calibration chosen from demo shots or probe submissions.**

**Final-phase use (3 submissions).**

1. The G2-best DIII-D blend plus the best rule-3-clean MAST path.
2. The same DIII-D blend with the runner-up MAST variant, nominated for Award #2 if Development scores favour it. The
   DIII-D denominator is the best across Final submissions, so a weaker DIII-D file costs nothing.
3. Held back as insurance against a platform fault or a failed rebuild check.

**Next sweep:** a delta sweep before the Final opens (2026-10-18), or sooner if a competitor posts a method.
