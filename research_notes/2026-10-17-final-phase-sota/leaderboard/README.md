# Development leaderboard, 2026-10-17 ~11:45 UTC

- [`dev_leaderboard_2026-10-17.txt`](dev_leaderboard_2026-10-17.txt): all 146 rows from
  `GET /api/phases/29580/get_leaderboard/?page=1..3` (best row per team, as the API returns it).
- [`per_scalar_2026-10-17.txt`](per_scalar_2026-10-17.txt) and [`details_parsed.json`](details_parsed.json): q95,
  βN and the seven consistency scalars per machine, parsed from each row's public Detailed Results page
  (`raw/details/<submission id>.html`). Parser: a one-off regex over the HTML. The `wstang` MAST row (S = 5.0) is a
  parse artefact; ignore it.
- `raw/page{1,2,3}.json`: the API responses.

## DIII-D: where the 0.0024 gap to the top sits (ours = submission 995469)

| term | ours | xiaoxiao5503 (0.996998) | argonclouds (0.996946) | weight x gap to #1 |
|---|---|---|---|---|
| li | 0.9544 | 0.9751 | 0.9763 | 0.2/7 x 0.021 = 0.00059 |
| κ | 0.9702 | 0.9847 | 0.9852 | 0.00042 |
| V | 0.9871 | 0.9948 | 0.9933 | 0.00022 |
| R_axis | 0.9823 | 0.9868 | 0.9852 | 0.00013 |
| Z_axis | 0.9912 | 0.9952 | 0.9946 | 0.00011 |
| δ_top / δ_bot | 0.9927 / 0.9916 | 0.9960 / 0.9955 | 0.9959 / 0.9953 | 0.00020 |
| q95 / βN | 0.9960 / 0.9959 | 0.9988 / 0.9986 | 0.9988 / 0.9981 | 0.15 x 0.0027 = 0.00041 |
| D_LCFS | 0.0097 | 0.0073 | 0.0067 | 0.1 x 0.0024 = 0.00024 |
| R²ψ | 0.9998 | 0.9999 | 0.9999 | ~0.00005 |

Consistency is 0.0017 of the 0.0024 gap; li and κ alone are 0.0010. Every top-15 team has li >= 0.966 and κ >= 0.978.

## MAST: the compliant-looking band and ours

Rows with MAST Consistency above ~0.85 (yszhang 0.951, dmitryzhevn 0.938, catskills 0.884; also dremovd 0.896 with
Consistency 0.73) are beyond any label-free method published. Development entries are not judged under rule 3, so
these say nothing about the Final. The only fully disclosed rule-3-clean pipeline is Anulum's (0.7145; their best
clean build 0.7190).

| team | MAST S | R²ψ | q95 R² | βN R² | D_LCFS | Cons | R_axis | κ | V | li |
|---|---|---|---|---|---|---|---|---|---|---|
| aidt_ipp | 0.8110 | 0.979 | 0.816 | 0.502 | 0.161 | 0.449 | 0.61 | 0.57 | 0.69 | 0.64 |
| kim | 0.8106 | 0.989 | 0.732 | 0.383 | 0.150 | 0.491 | 0.66 | 0.64 | 0.72 | 0.39 |
| argonclouds | 0.8067 | 0.989 | 0.758 | 0.359 | 0.151 | 0.471 | 0.70 | 0.56 | 0.77 | 0.20 |
| yangyiliu | 0.8028 | 0.986 | 0.701 | 0.325 | 0.152 | 0.495 | 0.67 | 0.42 | 0.71 | 0.34 |
| andregale ("Thomson-constrained GS") | 0.7824 | 0.987 | 0.638 | 0.305 | 0.175 | 0.432 | 0.66 | 0.73 | 0.66 | 0.48 |
| anulum (disclosed, clean) | 0.7145 | 0.979 | 0.456 | 0.206 | 0.192 | 0.227 | 0.08 | 0.43 | 0.58 | 0.14 |
| **chaleong** | 0.6406 | 0.917 | ~0.22 mean | | 0.222 | 0.127 | 0.33 | -0.24 | 0.33 | -0.28 |
| median of 43 rows with 0.6 < S < 0.87 | 0.729 | 0.980 | 0.414 mean | | 0.187 | 0.272 | | | | |

Our largest MAST deficits against the median, weight x gap: R²ψ 0.55 x 0.063 = **0.035**; Consistency
0.2 x 0.145 = **0.029**; q95/βN 0.15 x 0.19 = **0.029**; D_LCFS 0.1 x 0.035 = 0.004. Against the 0.80-0.81 cluster
the gaps double.

## Published code and method names (fact sheets)

| team | Dev rank (D3D) | link / method name | notes |
|---|---|---|---|
| anulum fortis | 11 (0.996359) | [github anulum/fusion-equilibrium-challenge](https://github.com/anulum/fusion-equilibrium-challenge), [HF weights](https://huggingface.co/anulum/fusion-equilibrium-challenge) | AGPL-3.0. Full methods report and experiment ledger, both machines |
| mohhakami | 13 (0.996223) | "F17K: F16K with Ip-sign canonical register / X-point heads on every shot"; [github MuhammadHakami/fusion-equilibrium](https://github.com/MuhammadHakami/fusion-equilibrium) | MIT. DIII-D only |
| pshishkin | 14 (0.996016) | [fork with experiments_mast.md](https://github.com/pshishkin/fusion-equilibrium-challenge-starter/blob/main/experiments_mast.md) | MAST notes use demo labels (pre-rule) |
| catskills | 37 | `https://github.com/catskillsresearch/catskills_fec` | 404 (private or deleted). Method name "03_D3D_unshifted_MAST_affine_betaN" |
| andregale | 19 | `https://github.com/andreaenea0` (account has 0 public repos) | "Thomson-constrained Grad-Shafranov + learned DIII-D ensemble"; MAST 0.782 |
| s_boehler | (MAST only) | "MAST 029: kinetic GS, vertical class from P6 radial-field magnitude + temporal smoothing" | MAST 0.758, Cons 0.33. Hint: up/down placement from the P6 radial-field coil |
| komakrv | 54 | "PCA-MLP (DIII-D) + zero-shot GS (MAST), sol 0.5 re-solved" | the solenoid half-factor again |
| broker, xexpertai, tariqj | - | Green's-function vacuum flux + learned residual; [tariqj repo](https://github.com/tariqjamil-bwp/fec-practice-17456) | all MAST S < 0.48; nothing to import |
| 19algorythms | - | [zenodo 23429908](https://zenodo.org/records/23429908) | embargoed to 2026-10-18; ridge axis + Gaussian well; low score |
| timelordraps, taxi347 | - | HF datasets of predictions | low score; predictions only |
