# Competition sources (fetched 2026-10-11)

Primary sources only: the organisers' paper, Codabench pages (via the public API), the official starter repo and
its issue threads (organiser answers by `ganatma`).

## Organiser paper: arXiv [2609.01750](https://arxiv.org/abs/2609.01750)

Nakkina, Sammuli, Hatch, Waller, Michoski, ... (Sophelio, General Atomics, UT Austin). Author-reported:

- Consistency = mean of 7 per-scalar R² (each clipped at 0): R_axis, Z_axis, κ, δ_top, δ_bot, V (2π∬R dR dZ inside
  the LCFS), l_i (ITER l_i(2)); κ/δ from LCFS bounding geometry, axis from the O-point. Failed derivations on the
  predicted map are mean-substituted.
- D_LCFS = symmetric Hausdorff / time-averaged true LCFS major radius, clipped at 1.
- Baselines are reported only on a 3-shot demo set (PCA+ridge SSIM 0.84; MLP MSE 0.005); "naive coil-mapping
  transfer" SSIM 0.83 DIII-D vs **0.10 MAST**. No composite baselines are published.
- Shot counts disagree between abstract (9,113 DIII-D / 2,416 MAST) and body (8,794 / 2,414).

## Codabench 17739 (API `/api/competitions/17739/`)

- Phases: Development 2026-07-27 to **2026-10-18 00:00 UTC** (5/day, 100 total); Final 2026-10-19 to
  **2026-10-26 00:00 UTC** (3 total, blind, leaderboard hidden).
- Forum: <https://www.codabench.org/forums/17429/> (login required; not readable from here).
- Ties broken by R²ψ, then D_LCFS.
- Award #2 uses the team's **best DIII-D S across Final submissions** as the denominator; a different Final
  submission can be nominated for each award by email before close.
- Honorable mentions (best R²ψ, R²{q95,βN}, Consistency, D_LCFS) are judged on the **MAST** private fold.

### Rule 3 (effective 2026-09-23; feedback clause 10-01; probe and mask clauses 10-04)

Barred for anything producing MAST predictions: MAST/MAST-U equilibrium reconstructions from any source (FAIR-MAST,
mirrors, derived datasets, **model weights trained on them**), for training, calibration, validation, selection,
hyperparameter search or checkpoint selection. Includes aggregates (bases, priors).

- The three labelled demo shots: format inspection and visualisation **only**.
- `fusion_scoring/masks/mast_envelope.npz`: only inside the official scoring derivation; never a model input,
  feature, loss mask or prior.
- Development-phase platform scores **may** be used to compare/tune/select MAST pipelines (disclose it).
- **Probes are not feedback**: constant/shifted/rescaled submissions built to measure labels, and any constant fitted
  to what they reveal, count as MAST equilibrium data.
- Permitted: MAST inputs (self-supervised pretraining, normalisation, distribution alignment, label-free test-time
  adaptation), MAST geometry/wall/coil descriptions, other machines' equilibria, synthetic equilibria from your own
  free-boundary solver calibrated against MAST *inputs* only.

### Rule 6

Reproduction package with every Final submission: source, pinned env, weights, from-scratch training entrypoint,
inference entrypoint, per-machine data provenance. Training must rebuild in **≤24 GPU-hours on one 80 GB A100**; CPU
data generation is uncapped but must be declared. Top five per challenge are rebuilt.

## Starter repo: [Sophelio/fusion-equilibrium-challenge-starter](https://github.com/Sophelio/fusion-equilibrium-challenge-starter)

- Only the sign of ψ is normalised (one bit per machine); amplitude is not.
- `np.round(psi, 3)` keeps R²ψ at 0.99997 but destroys ~35% of the MAST Consistency term; float16 is the
  recommended container ("costs ~0.1% of score").
- Vessel currents: reconstructions include them; the dataset omits them "for simplicity".
- Detailed Results report per-scalar consistency R² and the **derivation-failure rate** per machine.
- [#8](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/8) "zero-shot means no target-machine
  labels, not frozen parameters": label-free test-time updates such as physics-residual optimisation are allowed.
- [#9](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/9): DIII-D and MAST files may come from
  different pipelines (disclose the pairing).
- [#10](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/10): another team generates
  synthetic MAST equilibria with FreeGSNKE (~10 h on 64 CPU cores); organiser: CPU generation uncapped, permitted if
  driven by MAST inputs only and not calibrated on MAST EFIT.

## Development leaderboard snapshot (API `/api/phases/29580/get_leaderboard/`, 2026-10-11 08:00 UTC)

50 teams, 41 with MAST predictions. Best row per team as the API returns it.

| team | D S | D cons | D D_LCFS | D qb | M S | M R²ψ | M qb | M D_LCFS | M cons | G |
|---|---|---|---|---|---|---|---|---|---|---|
| argonclouds | 0.9967 | 0.9881 | 0.0068 | 0.9984 | 0.7414 | 0.9743 | 0.500 | 0.173 | 0.239 | 0.7439 |
| heavy4evy | 0.9965 | 0.9881 | 0.0075 | 0.9981 | 0.7678 | 0.9844 | 0.511 | 0.167 | 0.332 | 0.7704 |
| yangyiliu | 0.9965 | 0.9877 | 0.0071 | 0.9980 | 0.7466 | 0.9740 | 0.474 | 0.167 | 0.282 | 0.7492 |
| gwangsan | 0.9963 | 0.9872 | 0.0079 | 0.9981 | 0.8044 | 0.9887 | 0.549 | 0.151 | 0.467 | 0.8074 |
| dennisli | 0.9958 | 0.9852 | 0.0094 | 0.9983 | 0.8538 | 0.9909 | 0.583 | 0.106 | 0.660 | 0.8574 |
| dmitryzhevn | 0.9964 | 0.9875 | 0.0084 | 0.9983 | 0.9375 | 0.9957 | 0.802 | 0.051 | 0.873 | 0.9409 |
| catskills | 0.9944 | 0.9808 | 0.0106 | 0.9963 | 0.8636 | 0.9986 | 0.271 | 0.044 | 0.890 | 0.8684 |
| jygao | 0.9930 | 0.9758 | 0.0111 | 0.9955 | 1.0000 | 1.0000 | 1.000 | 0.000 | 1.000 | 1.0070 |
| **chaleong** (rank 29) | 0.9939 | 0.9788 | 0.0110 | 0.9959 | 0.6278 | 0.9079 | 0.320 | 0.220 | 0.012 | 0.6316 |

Median of the 26 teams with 0.5 < MAST S < 0.85: R²ψ 0.974, qb 0.320, D_LCFS 0.199, consistency 0.223.

Reading it (our inference, not stated by anyone): jygao's MAST S = 1.0000 can only come from the archive the hidden
labels are drawn from, and the next two (D_LCFS 0.04-0.05,
consistency ~0.88) are far beyond anything label-free we know of. Development entries are not judged against rule 3
retroactively, but the Final rebuild will be, so the rule-3-plausible band is G ≈ 0.70-0.86. Nobody on DIII-D
separates on R²ψ (all 0.9998-0.9999); the top-5 spread is consistency (~0.988) and D_LCFS (0.007-0.008).
