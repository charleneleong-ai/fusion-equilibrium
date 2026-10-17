# Experiment card: Fusion Equilibrium Challenge entry

Status 2026-10-11, development phase. Official best DIII-D S **0.993927** (rank 29 of 50, HF commit `07cfd3ff`); MAST
label-free S 0.627752, G 0.631588 ([`codabench.json`](../results/codabench.json)). Method and results:
[`report/main.tex`](../report/main.tex); story: [`journey.md`](journey.md).

## Target and inputs (frozen)

| Item | Value |
|---|---|
| Task | psi(R,Z) on 65 x 65, q95, betaN per EFIT frame from coils, Ip, Thomson; `S = 0.55 R2psi + 0.15 R2{q95,betaN} + 0.10 (1 - D_LCFS) + 0.20 Consistency`, Ch2 `G = S_MAST / S_DIII-D` |
| Data | HF `Sophelio/fusion-equilibrium-challenge`: `diii_d_train` 7,041 shots, `diii_d_public_test` 874, `mast_public_test` 1,206; no MAST training set. Dataset revision not pinned in the repo |
| Held-out split | 200 training shots, shot level, seed 0 ([`Frames.split`](../fusioneq/cache.py), [`HeldOut`](../fusioneq/evaluation.py)); v11 trained on the 160 never-scored shots (closed) |
| Scoring shots | first 40 held-out: every local number in the sweep writeups |
| Never-scored shots | 160 = 40 validation + 120 spare; from 8 Oct, blend screens are paired deltas on 5 x 40 groups (spare 1-3, validation, scoring) plus pooled S on the 160 ([`screens.yaml`](experiments/fusioneq/screens.yaml)) |
| Official test | Codabench public test (DIII-D and MAST), 5 submissions/day; Final phase 19-26 Oct, 3 submissions, retrain within 24 A100-h |
| MAST labels | 3 demo shots, inspection only under rule 3; no term is fitted to them in the current entry; the alpha / frac shortlist used them (disclosed below) |

## Components of the current best and the candidate

Checkpoints are `~/fusion/results/<tag>` on the A100 host ([`settings.py`](../fusioneq/settings.py)); not in git.

| Component | Exact members | [`v21_27_grids3`](../configs/blends/v21_27_grids3.yaml) (`07cfd3ff`) | [`v21_27_grids3_w75`](../configs/blends/v21_27_grids3_w75.yaml) |
|---|---|---|---|
| Base MLP ensemble (15) | `v5_loss_mix_i03`, `v6_ensemble_region_i00`-`i03`, `v8_ensemble_seeds_i00`-`i03`, `v14_ramp_i00`-`i05` | base share 0.10, MLPs 0.9 of it | same |
| Coefficient trees | XGBoost `xgb_top32_full.json` on host `v5_loss_mix_i03` (loaded from its `.ubj` copy) | 0.1 of base | same |
| Shape-target hybrid (27) | `v16_normalize_i00`-`i02`, `v17_normalize_more_i00`-`i02`, `v18_normalize_six_i00`-`i05`, `v19_normalize_variants_i00`-`i05`, `v20_normalize_ramp_i00`-`i02`, `v21_normalize_seeds_i00`-`i05` | 0.10 | 0.05 |
| Shape trees | XGBoost `xgb_shape_v21s40_top32_full.json` on host `v21_normalize_seeds_i00` | 0.15 | 0.10 |
| Grid U-Nets (3, averaged) | `grid_v2` (48 ch / 15 ep), `grid_v2_s1` (seed 1), `grid_v3` (64 ch / 20 ep) | 0.65 | 0.75 |
| MAST | physics flux, alpha 2.0, frac 0.4, solenoid-shape correction ([`physics.py`](../fusioneq/physics.py)) | as `4c428222` | not built for MAST yet |

Official entry: HF dataset `chaleong/fusion-eq-predictions` commit `07cfd3fffbf78a0b012f384291e269d9b4318698`. The
git commit that built it is not recorded. Libraries in [`uv.lock`](../uv.lock): torch 2.14.0, xgboost 3.2.0, numpy
1.26.4, scikit-learn 1.9.1, scipy 1.17.1, wandb 0.30.0, autoresearch 0.38.0 (`d016420a`). A100 runs use the starter
kit's venv (`~/fusion/starter/.venv`, [`mise.toml`](../mise.toml)), whose versions are not recorded.

## Budget, failures included

| Stage | Count |
|---|---|
| MLP sweeps v1-v21 | 115 scheduled iters in 21 [`schedules`](../configs/schedules); 9 not completed as scheduled (v2: 3 cut, v4: 2 killed by triage, v15: 4 killed by triage, rerun as v15b) |
| Grid models | 6 started; grid_v4 lost to an OOM kill on 11 Oct (`bf2c30b`), retraining; grid_v3_ctx8, grid_v3_env8 re-running; grid_v2_gs closed |
| Blend builds, 11 Oct | 2 OOM-killed (`77ee6db`); every w75 build OOM-killed until the `.ubj` loader (`87bf7a2`, count not recorded) |
| Experiments with a verdict | 27 ([`verdicts.yaml`](experiments/fusioneq/verdicts.yaml)): 11 PROMOTE, 5 KEEP, 11 DISCARD |
| Blend screens | 12 ([`screens.yaml`](experiments/fusioneq/screens.yaml)), 6 submitted |
| Official rows | 35: 31 scored, 3 held back, 1 unscored; 17 distinct DIII-D files, 15 new bests |
| Compute | per member 18-46 min (shape MLPs), ~24 min (v11 MLPs); total A100-hours not summed |

## Filter criteria

| Gate | Rule | Fixed before results? |
|---|---|---|
| G1 | beat matched-seed baseline on 40 scoring shots by more than seed noise (0.0013 one model, ~0.0002 ensemble) | **No.** Noise measured in v6, after v5 was promoted on a lucky seed (0.9915 vs 0.9901-0.9902) |
| G2 | win most 40-shot groups of the 160 never-scored shots, then beat the official best | **No.** Added 8 Oct after smoothing (+0.00018 local, 0.992144 vs 0.992168 official) and v21 (+0.00007 local, -0.000004 official) failed to transfer; applied as stated to every grid entry since |
| MAST solenoid shape | none local | chosen by official feedback (`4c428222`) |
| MAST alpha / frac | 36-setting sweep | **shortlisted on the 3 demo shots** (demo S 0.537 -> 0.584), **disclosed**; kept on official feedback: `d32bdf0c` MAST S 0.604496 vs `501b7986` 0.6017 (eps cap 0.65 vs 0.7: 0.6017 vs 0.6004) |

## Not measured

- w75 officially (160 shots 0.994000, +0.000086, all five groups); grid_v3_ctx8 / env8 / grid_v4 on 160 shots.
- Shot-level discordant counts (five group deltas give sign-test p >= 0.031).
- Cause of MAST R2psi 0.908 (solenoid-proportional level; vessel currents untested) and of MAST consistency 0.0117.
- Cause of the DIII-D consistency gap (0.9788 vs ~0.988) and of q95 / betaN (0.995946 vs 0.998).
- Final-phase reproduction: retrain within 24 A100-h and a label-free MAST rebuild, end to end.
- Retrain-to-retrain noise of the blend on the official score: identical files rescore identically (`df98320e`'s file
  in `d32bdf0c`), but no retrained blend has been submitted.
