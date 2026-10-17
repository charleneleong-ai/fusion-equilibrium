# DIII-D: where the gap to the leader is, and why v8 is this family's ceiling

Best official DIII-D S is **0.992168** (v8 9-model mean, HF `df98320e`, rank 26 of 77); the leader is 0.9963. Local S is
on the 40 held-out scoring shots and has run ~0.0006 above official. Sweep-by-sweep detail is in
[`experiments/fusioneq/mlp/`](experiments/fusioneq/mlp/) and [`verdicts.yaml`](experiments/fusioneq/verdicts.yaml).

## Where the gap is

Weighting the leader's published components by the score's weights:

| term (weight) | ours | leader | S lost |
|---|---|---|---|
| R2psi (0.55) | 0.99977 | 0.9999 | ~0.0001 |
| q95 + betaN R2 (0.15) | 0.9960 | 0.998 | ~0.0003 |
| 1 - D_LCFS (0.10) | 0.9880 | 0.9918 | ~0.0004 |
| consistency (0.20) | 0.9705 | ~0.987 (implied) | ~0.0033 |

About 80% of the gap is consistency, weakest in kappa (0.958) and li (0.967).

## The representation is not the limit

The true flux maps projected onto the model's PCA and scored with the true q95/betaN:

| components | S | consistency |
|---|---|---|
| 64 | 0.9986 | 0.9959 |
| 192 (the model's) | 0.9988 | 0.9964 |

Perfect coefficients would beat the leader, so the gap is the network's coefficient errors, not the PCA.

## What was tried after v8

| idea | local | official / verdict |
|---|---|---|
| more members (v9: 13 / 17) | 0.99277 / 0.99272 | ✗ saturated |
| +160 never-scored shots (v11) | 0.99260 (mix of 18: 0.99283) | ✗ not data-limited |
| temporal smoothing, sigma 0.5 (chosen on validation shots) | 0.992955 (+0.00018) | ✗ 0.992144 vs 0.992168 officially |
| whole-shot model (v12, [`v12_temporal.md`](experiments/fusioneq/mlp/v12_temporal.md)) | 0.98874 (3 seeds) | ✗ -0.0035 |
| predicting the 7 consistency scalars directly | kappa 0.865, li 0.836 | ✗ vs 0.958 / 0.967 from the flux |
| wider / longer members (v13, [`v13_capacity.md`](experiments/fusioneq/mlp/v13_capacity.md)) | 0.98826 / 0.98955 (2 seeds; base 0.99081) | ✗ |
| XGBoost on the top 32 coefficients, blended into v8 at w=0.1 (chosen on validation) | +0.00042 validation, -0.00006 scoring | ✓ **0.992317** officially (+0.00015, consistency 0.9705 → 0.9713) |
| the same trees trained to convergence (~10k rounds) | +0.00002 over the 3000-round trees on both sets | ✗ below the bar |
| a second, differently configured tree model averaged in | ±0.00002 | ✗ |
| tree weight 0.2 on ramp frames, 0.05 elsewhere | +0.00004 validation, worse on scoring | ✗ |
| ramp-weighted members (v14, [`v14_ramp.md`](experiments/fusioneq/mlp/v14_ramp.md)) in the ensemble | +0.00044 validation | ✓ small: **0.992341** officially (+0.00002) |
| six more ramp-weighted members (v15) | -0.00035 validation | ✗ |
| repairing frames whose derived volume / Z_axis jumps (neighbour average) | -0.0006 validation at 0.9% of frames | ✗ blending two flux maps moves the boundary |
| shape-only target (v16 / v17, [`v16_normalize.md`](experiments/fusioneq/mlp/v16_normalize.md)) as a hybrid blend | +0.00041 validation, +0.00066 scoring (6 members, w = 0.5) | ✓ first change both sets agree on: **0.99283** officially (+0.00049) |
| twelve shape-target members (v18, [`v18_normalize_six.md`](experiments/fusioneq/mlp/v18_normalize_six.md)), w = 0.65 | +0.00023 validation, +0.00016 scoring | ✓ **0.992904** officially (+0.00007) |
| re-tuning the tree and hybrid weights jointly | the submitted 0.1 / 0.65 is the validation optimum | ✗ |
| 21 members (+ v19 variants + v20 ramp, [`v19_normalize_variants.md`](experiments/fusioneq/mlp/v19_normalize_variants.md)) + 0.1 x XGBoost on the shape target | +0.00021 validation, +0.00018 scoring | ✓ **0.993078** officially (+0.00017) |
| 27 members (v21, [`v21_normalize_seeds.md`](experiments/fusioneq/mlp/v21_normalize_seeds.md)), hybrid 0.75 | +0.00007 validation, +0.00008 scoring | ✗ 0.993074 officially (-0.000004) |
| q95 / betaN from the shape members' heads (0.64) or XGBoost on q95 / betaN | +0.000055 validation, -0.000033 scoring | ✗ |
| grid shape model - a FiLM U-Net on the 65 x 65 map ([spec](specs/2026-10-08-grid-shape-model-design.md)) - in the blend, plus trees on all frames | grid_v1 at 0.1: +0.0001 on 160 shots; grid_v2 at 0.5: +0.0009 | ✓ **0.993186** (v1), then **0.99352** (v2, +0.000334) |
| grid refiner probes on the held-out error (ridge on flux / shape PCA, trees) | <= +0.00002 validation | ✗ error is shot-specific ([spec](specs/2026-10-07-grid-refiner-design.md)) |
| shape trees on a second host (v21 seed 40), alone or averaged with v16's | +0.000023 validation, +0.00002 scoring | ✗ below the bar |

Smoothing is the cautionary one: +0.00027 on the validation shots and +0.00018 on the scoring shots, yet slightly below
the unsmoothed ensemble officially, so local gains of ~0.0002 are inside the noise of 40 + 40 shots. Direct scalar
prediction loses 0.06-0.13 on every term: going through the full flux map is a far better route to these scalars than
regressing them, so there is no accurate target to steer the flux maps toward.

## Where the consistency error is

Per frame, v8's derived quantities against the labels on all 80 held-out shots (17,037 frames):

| term | R2 | error share of the worst 1% of frames | R2 with those fixed |
|---|---|---|---|
| li | 0.9655 | 57% | 0.9851 |
| Z_axis | 0.9571 | 49% | 0.9781 |
| volume | 0.9701 | 47% | 0.9840 |
| tri_top | 0.9850 | 53% | 0.9930 |
| kappa | 0.9687 | 27% | 0.9771 |

The error is concentrated, not broad. The worst frames sit twice as often in the first or last 10% of a shot (39%
against 20%), at lower current (median |Ip| 815-854 kA against 990) and higher true li (1.29 against 1.03, which we
under-predict); two shots hold 144-173 of them each. The model fits flat-top well and ramp-up / ramp-down poorly, so
the next step trains with those frames weighted up.

## Choosing on 160 shots instead of 40

40 validation shots cannot resolve gains of ~0.0001: the 27-member blend gained +0.00007 on both 40-shot sets and
nothing officially. The 120 spare held-out shots were never trained on or chosen on, so spare + validation gives 160
honest shots for choosing, with the scoring shots still report-only. Blend components are cached per held-out frame,
so a weight costs a re-blend and a score, not inference.

| blend (members, coefficient trees, hybrid, shape-tree host, weight) | 160 shots | official |
|---|---|---|
| (21, 0.1, 0.65, v16, 0.1) - `7a56d52a` | 0.991687 | 0.993078 |
| (27, 0.1, 0.75, v16, 0.1) - `6525ba1d` | 0.991698 (+0.00001) | 0.993074 |
| (27, 0.1, 0.75, v21, 0.15) - best on 160 ([`v21_27_st15.yaml`](../configs/blends/v21_27_st15.yaml)) | 0.992054 (+0.00037) | built, not yet submitted |

The 160-shot set called the 27-member null that both 40-shot sets missed. Its best blend raises the shape-tree weight
to 0.15, but per 40-shot group the gain is concentrated: -0.00003 / +0.00006 / **+0.00121** on the spare thirds,
+0.00007 validation, +0.00004 scoring. The hardest group (S 0.987) carries it; the official test set is harder than
our scoring shots (0.99308 against 0.99375), so some of that may transfer.

The same 160 shots and groups also closed two more levers: the shape members' q95 / betaN heads (weight 0.25-0.64:
every group within +-0.00003) and 64-coefficient trees instead of 32 (within +-0.00001 in every group).

Leaving one shot out of the hardest group shows where its +0.0012 comes from: without shot 6844 the gain is +0.00009,
without any other it stays ~+0.0012. Shot 6844 is pathological for every blend (alone: S 0.64, D_LCFS 0.156 against
~0.01 typically, kappa R2 -13); heavier shape trees make it less bad (consistency 0.089 -> 0.159). Such shots are rare
- one of 160 has D_LCFS above 0.05, and dropping the three worst-boundary shots lowers the mean D_LCFS only 3% (0.00995
-> 0.00964) - so the remaining loss is spread over ordinary shots, not a few failures. The candidate is therefore a
small bet: neutral on typical shots, better on rare bad ones.

Gating the shape-tree weight on a label-free signal - the RMS gap inside the envelope between the shape-tree and
base flux, in envelope units - keeps 0.1 where they agree and raises it to 0.2 on the 10% of frames that disagree
most. On 160 shots it recovers the same +0.00035 (0.992042), but per group it is the same pattern (-0.00008 / +0.00008
/ +0.0013 on the spare thirds, -0.00005 validation, +0.00004 scoring): the gate finds shot 6844's frames, and ordinary
shots gain nothing.

Splitting the hybrid weight by region (inside the envelope / outside, where only R2psi counts) finds the submitted
uniform 0.65 optimal on 160 shots: a lower outside weight loses (0.5: -0.00006, 0: -0.00042) and 0.75 inside loses
too, so the shape target helps outside the envelope as well.

Re-choosing the base ensemble's members on 160 shots changes nothing above noise: dropping the v14 ramp members
+0.000026 (every group -0.000001 to +0.000067), v8 alone +0.000004, dropping v5 + v6 -0.000029, dropping v6 -0.000031.

Trees refitted on twice the data (every 2nd training frame; the GPU freed up) are the first change positive in every
group. In the submitted structure, both tree models 2x: +29 / +71 / +61 / +57 / +8 (x 1e-6; spare thirds,
validation, scoring), 160 shots +0.000052; the shape trees gain more from the data than from the v21 host (1x on the
same host: +0.000006). Stacked with 27 members, hybrid 0.75 and shape trees 0.15
([`v21_27_st15_2x.yaml`](../configs/blends/v21_27_st15_2x.yaml)): +4 / +168 / +1272 / +106 / +47, 160 shots +0.00043 -
still led by shot 6844, but positive in every group and better than the 1x version in each.

## Conclusion

Every change since v8 (members, data, losses, history, smoothing, a sequence model, direct scalars, capacity, and a
second model family) is within noise or worse, except a second model family: 0.1 x XGBoost trees blended in (+0.00015 officially), now the DIII-D file. On the leaderboard ranks 17-26 sit within
0.0007 of each other, so a real +0.0003 would be worth a place, but nothing found here clears the noise of the local
shots.
