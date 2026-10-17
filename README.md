# Fusion Equilibrium Challenge (NeurIPS 2026)

Entry for the [Fusion Equilibrium Challenge](https://fusion-equilibrium-challenge.sophelio.io/)
([Codabench 17739](https://www.codabench.org/competitions/17739/)): reconstruct the 65x65 poloidal
flux map psi(R,Z) plus q95 and betaN at every EFIT timestamp of a tokamak shot from coil currents and
Thomson-scattering profiles only. Challenge 1 scores DIII-D with
`S = 0.55 R2psi + 0.15 R2{q95,betaN} + 0.10 (1 - D_LCFS) + 0.20 Consistency`; Challenge 2 scores
zero-shot transfer to MAST as `S_MAST / S_DIII-D`, gated on `S_DIII-D >= 0.85`.

Development phase closes **2026-10-18** (5 submissions/day, 100 total). The blind Final phase runs
2026-10-19 to 10-26 (3 submissions) and requires a reproduction package: pinned commit, weights,
train and inference entrypoints, training within 24 A100-hours.

## Approach

psi is split into a vacuum term and a plasma term. The vacuum term is the poloidal flux of the
shipped coil rectangles (elliptic-integral Green's functions driven by the ampere-turn currents),
which is linear, exact, and defined on any machine from its geometry alone. Models predict the
plasma residual. Inputs per EFIT frame are the coil currents, plasma current, and Thomson Te/ne
profiles resampled onto fixed spatial stations so shots with different channel layouts share one
feature space.

The DIII-D submission is a fixed-weight blend ([`configs/blends/`](configs/blends)): residual MLPs on
PCA coefficients and XGBoost coefficient trees set the flux level, while a shape-target MLP ensemble,
shape trees and FiLM U-Net grid models supply the shape inside the scorer's envelope. MAST is
zero-shot and label-free: calibrated coil flux plus a Thomson-placed plasma current, with textbook
q95 / betaN. **How it got here, step by step and with plots: [`docs/journey.md`](docs/journey.md).**

## Layout

| Path | What |
|---|---|
| `fusioneq/data.py` | One shot -> aligned per-frame inputs (errata fixes applied), targets untouched |
| `fusioneq/greens.py` | Filament and rectangle Green's functions; vacuum flux per coil column |
| `fusioneq/prepare.py`, `fusioneq/cache.py` | Parallel caching of a config into flat `.npy` arrays, and the reader |
| `fusioneq/score.py` | The official composite on held-out shots, driven from cached frames (vendored scorer) |
| `fusioneq/baseline.py` | PCA + ridge on the plasma residual with ridge heads for q95 and betaN |
| `fusioneq/model.py`, `fusioneq/targets.py` | Residual MLP on PCA coefficients; the shape-only target |
| `fusioneq/grid.py` | FiLM U-Net grid shape model (`train-grid`) and seed ensembles |
| `fusioneq/blend.py` | The submitted blend: base, coefficient trees, shape hybrid, shape trees, grid models |
| `fusioneq/physics.py`, `fusioneq/equilibrium.py` | Zero-shot MAST flux and scalars; the free-boundary Grad-Shafranov solve |
| `fusioneq/cli.py`, `fusioneq/commands/` | CLI: `prepare`, `train`, `train-grid`, `blend`, `build`, physics checks |
| `scripts/` | W&B report, dashboard and progress plots (built on `autoresearch.wandb_report` / `autoresearch.render`) |
| `vendor/starter/` | The organizers' starter kit, unmodified (scorer, errata fixes, baselines, submission tooling, dFL plugin) |

## Experiments and reporting

- **W&B**: every `baseline` / `train` run logs to `chaleong/fusion-equilibrium` (per-epoch loss and holdout R2,
  official composite and the seven consistency R2 values in the summary). `WANDB_MODE=disabled` turns it off.
- **autoresearch**: sweeps are schedules in `configs/schedules/*.yaml` run by `experiments/autoresearch.py`, which
  launches one `fusioneq.cli train` per iter under GPU triage and appends a BASELINE/KEEP/DISCARD/CRASH row to
  `experiments/fusioneq/<config>/results.jsonl` with the W&B URL. `autoresearch-report` scaffolds the sweep writeup
  under `docs/experiments/fusioneq/<config>/<schedule>.md`.
- **Experiment log**: [`docs/experiments/fusioneq/verdicts.yaml`](docs/experiments/fusioneq/verdicts.yaml) holds one
  row per experiment (baseline, promotion gate, result, verdict, PR). `scripts/update_wandb_report.py` renders it into
  the W&B report, `scripts/build_dashboard.py` into `report/dashboard.html`, and `scripts/progress_plots.py` draws the
  official progress plots under `experiments/progress/official/`.
- **Methods report**: `report/main.tex` (NeurIPS style); `scripts/report_table.py` refreshes its results table from
  `results/*_score.json`; `mise run build` inside `report/` produces the PDF.

## Workflow

Data (98 GB) and compute live on the A100 host under `~/fusion/` (`data/`, `cache/`, `results/`,
`submission/`, `starter/` with its venv). The code is synced there with `mise run sync`.

```bash
mise run install
mise run test
mise run sync
mise run remote -- prepare --config diii_d_train      # once per config, ~1 shot/s on 14 cores
mise run remote -- baseline --holdout 200              # fit + official score on held-out shots
mise run remote -- build --configs diii_d_public_test,mast_public_test

# a sweep, detached on the A100, then its writeup
ssh pi-a100-80gb 'cd ~/fusion/repo && setsid nohup ../starter/.venv/bin/python -m experiments.autoresearch run \
  --schedule configs/schedules/v1_mlp.yaml --config mlp --python ../starter/.venv/bin/python \
  </dev/null >>~/fusion/logs/sweep_v1_mlp.log 2>&1 & disown'
mise run fetch && uv run autoresearch-report --schedule configs/schedules/v1_mlp.yaml --config mlp --tag fusioneq \
  --experiments-dir experiments --out docs/experiments/fusioneq/mlp/v1_mlp.md
```

Submitting: push the two `.npz` to the private HF dataset repo `chaleong/fusion-eq-predictions`
with `vendor/starter/push_predictions.py` and a fine-grained read token scoped to that repo, then
upload the pointer zip on Codabench.
