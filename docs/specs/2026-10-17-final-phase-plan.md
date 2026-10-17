# Final phase plan (17 Oct)

The blind Final runs 19-26 Oct with 3 submissions. Development closes 18 Oct 11:00 AEDT. The organisers'
[23 Sep forum post](https://www.codabench.org/forums/17429/2807/) sets the terms:

- **Only the Final decides awards.** The Development board is practice and is not judged retroactively.
- **Rule 3 binds the Final.** No MAST or MAST-U equilibrium reconstructions, or aggregates or checkpoints derived from
  them, anywhere in the MAST path, including selection. Our own submissions and their public-fold scores are allowed
  (self-distillation). Synthetic equilibria from our own solver, calibrated on MAST inputs only, are allowed.
- **Every Final submission needs a reproduction package reference** in the fact sheet's Code URL field: a public
  repository at a pinned SHA, or a public archive with its sha256. The top five per challenge are rebuilt from the
  training entrypoint and re-scored, and the shipped weights must reproduce the submitted predictions. An entry that
  cannot be rebuilt is not eligible for an award.
- **The package contains:** model source, a pinned environment, the trained weights, a training entrypoint, an inference
  entrypoint, a data-provenance statement per machine, and the declared training compute (hardware and wall-clock,
  at most 24 GPU-hours on one 80 GB A100). The top three per award must also release the source under an OSI licence
  and file a 1-2 page methods report. The report must disclose the AI-assistant workflow.
- **Tie-breaks:** R2psi first, then D_LCFS (lower is better).
- In the Final, the per-shot table in Detailed Results is withheld; aggregates and the scalar breakdown remain.

## Where we stand

| | Official | Rank (17 Oct, 145 teams) | Rank 20 |
|---|---|---|---|
| Ch1, DIII-D S | 0.994423 (4bef7fb6); 0be92e18 pending, expected ~0.9946 | ~35 | 0.9958 |
| Ch2, G | 0.623 (688a773d MAST, rule-3 compliant) | ~37 | 0.7585 |

The Final configuration is 0be92e18:
[`v21_16_grids3_fno2`](../../configs/blends/v21_16_grids3_fno2.yaml) (16 MLPs, 2 forests, `grid_v3` and two FNO seeds at
0.75; 16.0-18.8 idle A100-h). It is built on the GPU in float32 with TF32 off (`NVIDIA_TF32_OVERRIDE=0`), stored with
`build --psi-dtype float32`, and gated on [`scripts/axis_jitter.py`](../../scripts/axis_jitter.py). The MAST file is
688a773d's.

Against the teams ranked about 20th, the Ch1 gap is mostly consistency:

| Term | Ours | Rank ~20 | Weight x gap |
|---|---|---|---|
| Consistency | 0.9808 | ~0.985 | +0.0008 |
| q95/betaN R2 | 0.9959 | ~0.9978 | +0.0003 |
| D_LCFS | 0.0100 | ~0.0082 | +0.0002 |
| R2psi | 0.9998 | 0.9999 | +0.0001 |

## A. Third FNO seed (17-18 Oct)

The second FNO seed was the largest blend gain measured (+186e-6 on the 160 shots). `grid_fno_s2` (grid_fno's recipe,
seed 2) started on the A100 at 17:01 AEDT 17 Oct (`~/fusion/chain_fno_s2.sh`). It takes about 3 h on an idle GPU.

1. Predict its held-out shapes in float32 and add it to the 160-shot screen
   (`~/fusion/final_budget/subsets160_fno.py`).
2. Screen B15 and B7 bases with `grid_v3` plus three FNOs, at grid weight 0.75 and 0.80. 0.75 was the edge of the last
   screen.
3. Gate (fixed now, before results): beat `v21_16_grids3_fno2` by more than 1e-4 on the 160 shots, win at least 4 of 5
   groups, and keep the upper hours bound at 21 h or below (3 h margin).

| Blend | Idle A100-h |
|---|---|
| B15 + v3 + 3 FNOs | 19.0-22.1 (over the margin) |
| B7 + v3 + 3 FNOs | 16.8-19.3 |

If a blend passes, build it on the GPU in float32 with float32 psi and submit it to Development before the 11:00 close,
as one change against 0be92e18. The screen has transferred at about 0.5 for blend changes.

## B. Reproduction package (18-20 Oct)

- **One training entrypoint**, `fusioneq rebuild --spec <blend>`, that retrains every member from its stored recipe
  (`config.json` for MLPs, `grid_config.json` for grids, the tree settings for forests), then runs `blend` and
  `build --psi-dtype float32`. Before relying on it, check each stored config holds every flag its member was trained
  with. Reuse [`RebuildBudget`](../../fusioneq/budget.py) for the dry run that lists the jobs and hours.
- **The MAST pipeline as an entrypoint too.** 688a773d's file comes from `make_mast_shift_probe.py` (shifted flux plus
  22d99887's mask-free scalars), and it must be rerun on the Final MAST inputs, so it moves into the CLI.
- **Public reference.** `charleneleong-ai/neurips-workshops-2026` is private, so the package goes in a new public
  repository holding only `competitions/fusion-equilibrium` (MIT licence), cited at a pinned SHA. The weights go in a
  public HF model repo, cited by revision.
- **Environment:** `uv.lock`, plus the GPU settings (`NVIDIA_TF32_OVERRIDE=0`, float32 prediction).
- **Data provenance:**
  - DIII-D: the challenge train split only.
  - MAST: the shipped inputs and the machine geometry only. No MAST equilibria; the demo shots were used only for
    inspection.
  - Disclose that the MAST pipeline choices were compared on our own public-fold scores.
- **Methods report** (1-2 pages), including the AI-assistant workflow.

## C. From-scratch rehearsal (start by 19 Oct)

Run the rebuild of the Final pick on the A100, timed with the GPU sampler. It passes if it finishes inside 24 h and the
rebuilt blend scores within 2e-4 of the original on the 160 shots (about seed noise). Retraining has never been run end
to end. The shared GPU can stretch wall time, so record busy and idle hours separately.

## D. Final submissions (3)

1. The Final pick (0be92e18's configuration) on the Final test inputs: the safe entry.
2. The three-FNO blend, if A passes.
3. Held back for a fix after the first scores.

## Ch2

MAST stays at 688a773d's pipeline. Teams with MAST R2psi of 0.98-0.999 and consistency up to 0.89 on the Development
board look as if they trained on MAST equilibria. The forum post bars that from the Final, so the compliant field there
is likely much narrower than the practice board shows. Every compliant MAST lever tried so far is closed
([`mast-diagnosis.md`](../mast-diagnosis.md)).

The post also allows public MAST inputs that are not in the release, such as FAIR-MAST level-1 signals. They cannot be
used on the test fold, because its shot identities are withheld and matching content to the archive is the leak the
organisers closed. The one open compliant route is synthetic MAST equilibria from our own free-boundary solver,
calibrated on MAST inputs only. That is a multi-day project and is not planned before the Final.

## Not established

- The idle time of an FNO seed is inferred from shared-GPU runs (3.0 h on 12 Oct), not measured on an idle GPU.
- Whether a from-scratch rebuild reproduces the trained members' scores.
- Whether "24 GPU-hours" counts GPU tree fitting. Inference is separate from training in the post.
- Whether float32 psi storage gains as much on the Final shots as on Development (+0.000386).
