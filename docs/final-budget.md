# Final-phase rebuild budget (DIII-D)

Measured 2026-10-12 / 13 (AEDT). The submitted best, 07cfd3ff
([`configs/blends/v21_27_grids3.yaml`](../configs/blends/v21_27_grids3.yaml)), needs **20.6-26.9 idle A100-hours** to
train, so it may not fit the Final's 24-hour cap. Cutting the PCA shape hybrid from 27 networks to its shape-tree host
([`configs/blends/v21_16_grids3.yaml`](../configs/blends/v21_16_grids3.yaml)) brings the estimate to **13.4-17.8 h** and
scores **0.993941** on the 160 never-scored shots, against 0.993914 for 07cfd3ff.

## 15 Oct: the official losses were bf16 GPU builds

[`docs/fno-transfer-diagnosis.md`](fno-transfer-diagnosis.md): a4b4092c and 3facfca7 were built on the GPU, where grid
inference ran under bf16 autocast, and both lost on the magnetic axis only. Every CPU (float32) build transferred.
`GridPredictor.predict_shape` now runs in float32 on any device, so an A100 Final rebuild no longer pays this. The
float32 screen's rankings stand, including `v21_16_grids3` and the FNO blends; the sections below are kept as written.

## 14 Oct, later: the FNO pick is withdrawn (superseded by 15 Oct)

The FNO did not transfer. a4b4092c (07cfd3ff plus `grid_fno`) scored DIII-D S **0.993714** officially, 0.000213 below
07cfd3ff, where the 160-shot screen had it 0.000212 above; the loss is all consistency. The Final pick goes back to
[`configs/blends/v21_16_grids3.yaml`](../configs/blends/v21_16_grids3.yaml) (13.4-17.8 h). The section below is kept as the
record of the screen.

## 14 Oct update: two FNO grids change the pick (superseded)

Screened 13 Oct 22:58 to 14 Oct 06:13 AEDT, on CPU on the box. The new pick is
[`configs/blends/v21_16_grids3_fno2.yaml`](../configs/blends/v21_16_grids3_fno2.yaml). It keeps the 16-MLP base of
v21_16_grids3 and replaces its three CNN grids with `grid_v3` and two FNO seeds, `grid_fno` and `grid_fno_s1`, at grid
weight 0.75.

- It trains in **16.0-18.8 idle A100-h**.
- It scores **0.994433** on the 160 never-scored shots. That is +519e-6 over 07cfd3ff and +307e-6 over a4b4092c
  ([`v21_27_grids3_fno`](../configs/blends/v21_27_grids3_fno.yaml), the submitted one-FNO blend).
- It beats 07cfd3ff on all five 40-shot groups. Against a4b4092c it wins four groups and is -8e-6 on the 40 scoring
  shots, which is a tie.

a4b4092c itself needs 23.6-30.2 h, so it cannot be the Final.

The fallback, [`configs/blends/v21_8_grids3_fno2.yaml`](../configs/blends/v21_8_grids3_fno2.yaml), cuts the base to
v5_i03 + v14. It needs 13.8-16.0 h and scores 0.994416.
[`configs/blends/v21_16_grids3_fno.yaml`](../configs/blends/v21_16_grids3_fno.yaml) is the best blend with one FNO
(s1+v3+fno, 14.7-18.3 h, 0.994255), for use if the second seed is left out.

The screen script is `~/fusion/final_budget/subsets160_fno.py`, run by `run_fno.sh` and then `run_fno2.sh`. It reuses
the per-family sums from the 12 Oct screen and the cached held-out FNO shapes. `grid_fno_s1` was predicted on CPU from
`~/fusion/repo_fno` into `~/fusion/cache/final_budget/grid_fno_s1_fp32.npy`. The screen reproduces a4b4092c at 0.994126
and v21_16_grids3 at 0.993941. It scored two passes:

- **One FNO (54 blends):** bases B15, B7 or B1; shape S1; both forests kept; grid sets v2+s1+v3+fno, v2+s1+v3,
  s1+v3+fno, v2+v3+fno, v2+s1+fno and v3+fno; grid weight 0.65, 0.70 or 0.75.
- **Two FNOs (12 blends):** bases B15 or B7; grid sets v2+s1+v3+fno+fno1, s1+v3+fno+fno1 and v3+fno+fno1; grid weight
  0.70 or 0.75.

S1 keeps only `v21_normalize_seeds_i00`, because it hosts the shape trees. Each FNO seed is costed at 3.0-3.3 h, now in
[`GRID_HOURS`](../fusioneq/budget.py#L15-L21). `grid_fno` ran 3.0 h on 12 Oct. `grid_fno_s1` took 5.6 h wall on the
shared GPU (11:33-17:09 UTC, 13 Oct), at 8-17 min per epoch depending on the other job.

The rows are in [`final_budget_subsets160_fno.json`](experiments/fusioneq/final_budget_subsets160_fno.json) (one FNO)
and [`final_budget_subsets160_fno2.json`](experiments/fusioneq/final_budget_subsets160_fno2.json) (two FNOs). The group
columns give the difference from 07cfd3ff in units of 1e-6.

| Blend | MLPs | Idle A100-h | 160 shots | spare0 | spare1 | spare2 | validation | scoring |
|---|---|---|---|---|---|---|---|---|
| B1/S1/v3+fno @0.75 | 2 | 9.2-10.6 | 0.993983 | +76 | -325 | +460 | -61 | -67 |
| B7/S1/v3+fno @0.75 | 8 | 10.8-12.7 | 0.994227 | +149 | -176 | +785 | +369 | -44 |
| B7/S1/s1+v3+fno @0.75 | 8 | 12.5-15.5 | 0.994234 | +143 | +43 | +709 | +350 | +40 |
| **B7/S1/v3+fno+fno1 @0.75 (fallback)** | 8 | **13.8-16.0** | **0.994416** | +352 | +48 | +951 | +582 | +92 |
| B15/S1/s1+v3+fno @0.75 (best with one FNO) | 16 | 14.7-18.3 | 0.994255 | +180 | +112 | +708 | +338 | +61 |
| **B15/S1/v3+fno+fno1 @0.75 (recommended)** | 16 | **16.0-18.8** | **0.994433** | +382 | +116 | +937 | +576 | +112 |
| B15/S1/s1+v3+fno+fno1 @0.75 | 16 | 17.7-21.6 | 0.994410 | | | | | |
| B15/S1/v2+s1+v3+fno+fno1 @0.75 | 16 | 19.4-24.4 | 0.994316 | | | | | |
| B15/S1/v2+s1+v3 @0.65 (v21_16_grids3) | 16 | 13.4-17.8 | 0.993941 | -20 | +12 | +82 | +34 | +60 |
| B15/S27/v2+s1+v3+fno @0.70 (a4b4092c) | 42 | 23.6-30.2 | 0.994126 | +157 | +83 | +339 | +251 | +120 |
| B15/S27/v2+s1+v3 @0.65 (07cfd3ff) | 42 | 20.6-26.9 | 0.993914 | 0 | 0 | 0 | 0 | 0 |

The rows marked as picks are the hours frontier. The last three are references, from the 12 Oct screen and from
`grid160k.py`.

What the 66 rows show:

- **The second FNO seed is the biggest lever.** B15/S1/v3+fno 0.994247 rises to 0.994433 with `grid_fno_s1`, a gain of
  +186e-6. Against the best one-FNO blend, the recommended blend gains on all five groups: +202 / +4 / +229 / +238 /
  +51.
- **Fewer CNN grids are better once the FNOs are in.** With two FNOs, adding `grid_v2_s1` costs 23e-6 and adding both
  48-channel grids costs 117e-6. With one FNO, `grid_v2` was the weak seed and `grid_v2_s1` a tie-breaker.
- **Grid weight 0.75 wins in every base and grid set,** by +10e-6 to +60e-6 over 0.65. At 0.75 the shape MLP's hybrid
  share is zero, so the network is trained only as the shape-tree host. 0.75 is the edge of the screened range. Going
  past it would take weight from the shape trees or the base, and that was not tried.
- **B7 is within 17e-6 of B15** with the same grids. It still wins all five groups against 07cfd3ff and saves 2.2-2.8 h.

Against a4b4092c, the recommended blend is +225 / +33 / +598 / +325 / -8 x 1e-6 on the five groups.

**Caveats specific to this pick:**

- **Post hoc.** The grid set, the weight and the second FNO seed were all chosen on the same 160 shots they are reported
  on. None of these numbers was fixed before the results were seen.
- **The FNO itself is unconfirmed.** The official score of a4b4092c decides whether an FNO grid belongs in the Final at
  all. If a4b4092c does not beat 07cfd3ff on Codabench, fall back to
  [`v21_16_grids3`](../configs/blends/v21_16_grids3.yaml). `grid_fno_s1` has no external score. A Development
  submission of v21_16_grids3_fno2 before the 18 Oct close would test it and the 0.75 / no-48-channel choices together.
- **Idle FNO time is inferred, not measured.** Both seeds trained on a shared GPU. The 3.0-3.3 h figure assumes the
  first seed's 3.0 h was close to idle.

## The rule

Rule 6, as transcribed from the Codabench pages in
[`research_notes/2026-10-11-fusion-equilibrium-sota/competition.md`](../research_notes/2026-10-11-fusion-equilibrium-sota/competition.md#L44-L48):

> Reproduction package with every Final submission: source, pinned env, weights, from-scratch training entrypoint,
> inference entrypoint, per-machine data provenance. Training must rebuild in **≤24 GPU-hours on one 80 GB A100**; CPU
> data generation is uncapped but must be declared. Top five per challenge are rebuilt.

The cap is on training. Inference and CPU data preparation are listed separately below and are declared, not counted.
MAST is physics only with the DIII-D-fitted [`shape_map.py`](../fusioneq/shape_map.py), so it trains nothing.

## Training hours per component

"As run" is wall time on the shared A100, where another project often held the GPU. "Idle" is the estimate for one
unshared A100. Sources: `results/<tag>_score.json` `runtime_min` (the same as W&B `_runtime` in project
`chaleong/fusion-equilibrium`), per-epoch timestamps in `~/fusion/logs/grid_*.log`, and the tree chain log
`~/fusion/logs/chain_trees_more.out`.

| Component | Count | As run (h) | Idle A100 (h) | Evidence |
|---|---|---|---|---|
| Base MLPs (v5 / v6 / v8 / v14) | 15 | 6.4 | 4.1-5.3 | 16.5-46.1 min each; sweeps ran one at a time. The idle rate of 16.5-21 min per network comes from v14_ramp_i01-i05 (16.4-16.9 min) and the v17 rerun on a free GPU (17.0-17.8 min, against 41-65 min shared). Times include held-out scoring and W&B logging. |
| Shape MLPs (v16-v21) | 27 | 11.6 | 7.4-9.5 | 16.7-46.4 min each, same architecture (10 × 2048, 120 epochs) |
| XGBoost forests (`xgb_top32_full`, `xgb_shape_v21s40_top32_full`) | 2 | ~2.5 | 2.0-2.6 | GPU fit started 05:11 UTC 8 Oct, after the chain waited for less than 20 GB of GPU use. The shape forest was saved 77 min later, model loading included. The coefficient forest is about the same size, 3000 rounds. |
| `grid_v2` (48 ch, 15 epochs) | 1 | 4.9 | 1.7-2.8 | 18.7 min per epoch while a duplicate `grid_v2` run (`quvbe78z`) shared the GPU |
| `grid_v2_s1` (48 ch, 15 epochs) | 1 | 6.1 | 1.7-2.8 | 19.1 min per epoch, plus 101 min to the end of the first epoch |
| `grid_v3` (64 ch, 20 epochs) | 1 | 10.0 | 3.6-4.0 | 19-41 min per epoch shared. `grid_v3_env8` has the same size and ran 10.5 min per epoch on a free GPU. |
| `grid_fno` (FNO width 48, 20 epochs; 14 Oct) | 1 | 3.0 | 3.0-3.3 | 08:41-11:40 UTC, 12 Oct, about 9 min per epoch. Not in 07cfd3ff's total below. |
| **Total** | 42 MLPs + 2 forests + 3 grids | **~41.5** | **20.6-26.9** | |

The 48-channel grids never ran on an idle GPU. Their range runs from (48/64)² of the 64-channel rate (5.9 min per epoch)
up to that full rate (10.5 min per epoch), plus about 11 min of startup. Nothing measured them directly, so their range
is the widest.

Declared, not capped:

- CPU data preparation: 36 min for `diii_d_train` (7041 shots, 1.56 M frames) and 3-6 min for the public test
  (`~/fusion/logs/prepare_*.log`).
- Inference: building the full DIII-D public test on CPU took 2.8 h (`build_v21_27_grids3.log`: blend 19:15 UTC, build
  done 22:12 UTC, 10 Oct). It can also run on the GPU.

[`fusioneq rebuild-budget`](../fusioneq/commands/budget.py) prints these ranges for any blend spec. The constants are in
[`fusioneq/budget.py`](../fusioneq/budget.py).

```bash
uv run python -m fusioneq.cli rebuild-budget configs/blends/v21_27_grids3.yaml configs/blends/v21_16_grids3.yaml
# v21_27_grids3: 42 MLPs, 2 forests, grids [...]: 20.6-26.9 A100-h of 24 (at risk)
# v21_16_grids3: 16 MLPs, 2 forests, grids [...]: 13.4-17.8 A100-h of 24 (fits)
```

## Score against hours

The screen script, `~/fusion/final_budget/subsets160.py`, ran on CPU on the box from 21:04 to 23:08 AEDT on 12 Oct. It
cached per-family sums of every member's held-out psi and scalars under `~/fusion/cache/final_budget/`. Rebuilding the
15-member base from those sums matched the cached `ens.npy` to a maximum absolute difference of 6e-8. The script then
scored 80 blends, every combination of these options, using 07cfd3ff's weights. 07cfd3ff's own weights are base 0.10,
hybrid 0.10, shape trees 0.15 and grids 0.65.

- **Base:** B15 (all), B7 (v5_i03 + v14), B5 (v5_i03 + v8), or B1 (v5_i03 alone). v5_i03 is always kept because it
  hosts the coefficient trees.
- **Shape:** S27 (all), S12 (v16 + v17 + v21), S6 (v21), or S1 (v21_i00 alone). v21_i00 is always kept because it hosts
  the shape trees.
- **Grids:** all three, v2 + v3, v2 + v2_s1, v3 alone, or v2 alone.

All 80 results are in
[`docs/experiments/fusioneq/final_budget_subsets160.json`](experiments/fusioneq/final_budget_subsets160.json).

The table shows the frontier, meaning the best 160-shot score at or below each upper-bound hour count, plus 07cfd3ff.
The group columns give the difference from 07cfd3ff in units of 1e-6.

| Blend | MLPs | Idle A100-h | 160 shots | spare0 | spare1 | spare2 | validation | scoring |
|---|---|---|---|---|---|---|---|---|
| B1/S1/v2 | 2 | 4.3-6.1 | 0.992881 | -674 | -817 | -1743 | -931 | -372 |
| B1/S1/v3 | 2 | 6.2-7.3 | 0.993626 | -360 | -496 | +38 | -466 | -125 |
| B5/S1/v3 | 6 | 7.3-8.7 | 0.993828 | -272 | -285 | +323 | -217 | -99 |
| B7/S1/v3 | 8 | 7.8-9.4 | 0.993887 | -271 | -348 | +391 | -19 | -93 |
| B15/S1/v3 | 16 | 10.0-12.2 | 0.993914 | -236 | -270 | +411 | -32 | -76 |
| B7/S1/3 grids | 8 | 11.2-15.0 | 0.993919 | -45 | -65 | +68 | +52 | +57 |
| **B15/S1/3 grids (recommended)** | 16 | **13.4-17.8** | **0.993941** | -20 | +12 | +82 | +34 | +60 |
| B15/S27/3 grids (07cfd3ff) | 42 | 20.6-26.9 | 0.993914 | 0 | 0 | 0 | 0 | 0 |

What the 80 rows show:

- **The 27-network shape hybrid adds nothing at 0.10 weight.** With the base and grids fixed, S1 scores the same as or
  better than S27 on the 160 shots in every base row (B15: 0.993941 against 0.993914). It accounts for 7.4-9.5 of the
  hours.
- **The grids are the lever.** Dropping `grid_v2` and `grid_v2_s1` (B15/S1/v3) ties on the 160 shots but loses on 4 of
  the 5 groups, by up to 270e-6. Keeping only `grid_v2` costs about 780e-6.
- **Base size matters little once the grids are in.** B7 is within 22e-6 of B15 on the 160 shots and saves 2.4 h.

## Recommendation (12 Oct, before the FNO; superseded above)

Use [`configs/blends/v21_16_grids3.yaml`](../configs/blends/v21_16_grids3.yaml) for the Final rebuild: 07cfd3ff with
`shape: [v21_normalize_seeds_i00]` and nothing else changed.

- It scores 0.993941 on the 160 shots and beats 07cfd3ff on 4 of the 5 40-shot groups.
- It trains in 13.4-17.8 idle A100-hours, leaving at least 6 h of margin for the 48-channel grids' uncertainty and for
  retraining on all shots.
- If the margin has to grow, B7/S1/3 grids (0.993919, 11.2-15.0 h) is the fallback.

## 15 Oct: idle-GPU timing

`chain_timing.sh` waited for the A100 to sit idle (<2 GB, <5% util for 10 min), then timed one v21 MLP on the full
recipe and 3 epochs of the 48-channel grid (grid_v2 recipe). Nothing else used the GPU during either run.

| Job | Measured | Budget model |
|---|---|---|
| One MLP, 120 epochs, scoring included | 19.3 min (1159 s, 25 GB host RSS) | 16.5-21 min |
| 48-channel grid, first epoch (data setup included) | 19.0 min | |
| 48-channel grid, each later epoch | 8.3 min (40 GB GPU, 54 GB host RSS) | |
| 48-channel grid, 15 epochs (extrapolated) | about 2.25 h | 1.7-2.8 h |

Both fall inside the model's ranges, so `fusioneq/budget.py` stands. At these point values 07cfd3ff's blend (42 MLPs,
2 forests, grid_v2, grid_v2_s1, grid_v3) needs about 13.5 + 2.3 + 4.5 + 3.8 = 24.1 h: at the cap, with no margin. The
13.4-17.8 h pick above remains the safe Final choice.

## What is not established

- **Chosen after seeing results.** The pick was made on the same 160 shots it is reported on. Its +27e-6 lead over
  07cfd3ff is about the size of the gaps between groups, so the claim is "no worse at under 18 h", not "better".
- **No external score.** The recommended blend has not been scored on Codabench. One Development submission before the
  18 Oct close would check it against 07cfd3ff's leaderboard score.
- **MLP times are upper bounds.** They include held-out scoring and W&B logging. Training several MLPs at once on the
  one GPU might cut the total, but this was not measured.
- **A from-scratch rebuild has not been run.** It is unknown whether one reproduces these held-out numbers, given seeds,
  AMP nondeterminism, and retraining on the 200 held-out shots. The 200 held-out shots would add about 3% more frames.
- **The organisers' accounting is unconfirmed.** It is not known whether they count GPU tree fitting or GPU inference
  against the cap. The table counts tree fitting and excludes inference.
