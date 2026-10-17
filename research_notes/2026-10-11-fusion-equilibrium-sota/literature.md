# Literature and released code (2024-2026), fetched 2026-10-11

All numbers are author-reported, taken from the arXiv abstract page or the publisher record. "abstract only"
means the full text was not read.

## Magnetics-free / reduced-diagnostic reconstruction (closest to our task)

| work | machine | inputs | result (author-reported) | code |
|---|---|---|---|---|
| Stokolesov et al., [arXiv 2505.10709](https://arxiv.org/abs/2505.10709) (May 2025; J. Plasma Phys. 91 E152) | DIII-D, ~25k shots 2004-2024 | coil currents only; + Ip and loop voltage | LCFS mean point displacement 0.04 m (coils only), 0.03 m (+Ip, V_loop) | none listed |
| Sun et al., [Nucl. Fusion 64 (2024)](https://doi.org/10.1088/1741-4326/ad5d7b) ([OSTI](https://www.osti.gov/pages/biblio/2403389)) | DIII-D 2019 campaign, kinetic EFIT | magnetics, density/temperature, actuators | "magnetics-only model outputs accurate kinetic profiles"; extra data helps single-target (p, j) models most. Abstract only | none |
| Nakkina et al., [arXiv 2609.01750](https://arxiv.org/abs/2609.01750) (organisers) | DIII-D + MAST | coils, Ip, Thomson | demo-set baselines only (see competition.md) | starter repo |

Stokolesov's 3-4 cm mean displacement from coils is a *mean*, ours is a Hausdorff max (D_LCFS 0.011 × ~1.7 m
≈ 1.9 cm): the published reduced-diagnostic boundary work is behind the leaderboard, not ahead of it.

## Neural equilibrium surrogates (magnetics or synthetic data)

| work | setting | result (author-reported) | code |
|---|---|---|---|
| Grandin, Mele, Galperti et al., [arXiv 2606.09487](https://arxiv.org/abs/2606.09487) (CDC 2026) | TCV, ~10k shots, liuqe surrogate, branch (magnetics) / trunk (R,Z) with autodiff physics regularisation | <100 µs inference, real-time deployed, accuracy comparable to liuqe-rt | [matgrand/liuqe-nn](https://github.com/matgrand/liuqe-nn) |
| Zheng et al., [arXiv 2405.11221](https://arxiv.org/abs/2405.11221) (2024) | HL-3, 68 magnetic channels, 1,159 shots, 129×129 | mean R² 0.941 (8 scalars), 0.997 (Ψ), 0.959 (J_P) | none |
| EFIT-mini, [arXiv 2503.19467](https://arxiv.org/abs/2503.19467) (2025) | EXL-50U, 355 shots, NN inside EFIT | 0.11 ms NN / 0.36 ms full at 129×129 | none |
| Krastev, [arXiv 2608.05555](https://arxiv.org/abs/2608.05555) (Aug 2026) | FNO on FreeGS double-null free-boundary equilibria | 0.05% mean rel. L2; error ∝ N^-0.68 over 500-5000 train; X-point within 0.2 cm, O-point 0.03 cm | none |
| Ding et al., [arXiv 2511.19114](https://arxiv.org/abs/2511.19114) (Nov 2025) | PINO, LCFS shape + nonlinear current profile → ψ (appears fixed-boundary) | TKNO 0.25% rel. L2 supervised; physics loss cuts residual ~10^4×; semi-supervised extrapolation 4.76% (degradation 8.9× vs 39.8× supervised) | none |
| Yoo et al., [arXiv 2606.15512](https://arxiv.org/abs/2606.15512) (Jun 2026) | cross-device transfer, 8 synthetic Solov'ev geometries | WNO <4% rel. L2 with 100 target equilibria, <2% with full fine-tuning; single-geometry pretraining transfers poorly | none |
| Ross et al., [arXiv 2605.14939](https://arxiv.org/abs/2605.14939) (May 2026) | MAST-U shape emulators from >10^6 simulated GS equilibria | abstract gives no error numbers | none |
| Wang et al., [arXiv 2608.23976](https://arxiv.org/abs/2608.23976) (Aug 2026) | TCV inverse GS (boundary → coil currents) | experimental shaping demo; no numbers in abstract | none |

Physics-informed PINN solvers ([2311.13491](https://arxiv.org/abs/2311.13491), [2507.16636](https://arxiv.org/abs/2507.16636))
are verified only against analytic equilibria, not measured data.

## Cross-machine and MAST-specific

- No paper found that transfers a learned equilibrium model DIII-D → MAST (or any conventional → spherical tokamak)
  without target labels. Yoo et al. is the nearest (synthetic, needs ~100 target labels, which rule 3 forbids).
- TokaMark ([arXiv 2602.10132](https://arxiv.org/abs/2602.10132), 14 MAST tasks,
  [code](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark_baseline)) and TokaMind
  ([arXiv 2602.15084](https://arxiv.org/abs/2602.15084), [code](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind),
  weights on [HF UKAEA-IBM-STFC](https://huggingface.co/UKAEA-IBM-STFC)) are pretrained on FAIR-MAST. The abstracts do
  not say whether EFIT signals were included; if they were, rule 3 bars the weights ("model weights trained on" MAST
  equilibria). Treat both as off-limits unless their data cards prove otherwise.
- FreeGSNKE validated against MAST-U EFIT++ ([arXiv 2407.12432](https://arxiv.org/abs/2407.12432)); release
  [v3.1.1](https://github.com/FusionComputingLab/freegsnke) (2026-09-24). Ships MAST-U, not MAST, machine files.
- FreeGS ([freegs-plasma/freegs](https://github.com/freegs-plasma/freegs), `freegs/machine.py`) has `MAST()` /
  `MAST_sym()` (P2-P6 point coils, P1 solenoid R = 0.15 m, Z ±1.4 m) and `DIIID()`: a second, public MAST coil
  description to cross-check the solenoid geometry the challenge ships. Machine geometry is rule-3 permitted.

## Hugging Face

`hf://papers` search and Hub search ("fusion equilibrium") return only the challenge dataset
([Sophelio/fusion-equilibrium-challenge](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge));
no equilibrium-reconstruction checkpoints are public. No public competitor solution repos were found (GitHub hits
are starter-kit forks).
