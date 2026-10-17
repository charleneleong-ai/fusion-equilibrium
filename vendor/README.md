# vendor/

`starter/` is the organizers' starter kit, copied unmodified from
[Sophelio/fusion-equilibrium-challenge-starter](https://github.com/Sophelio/fusion-equilibrium-challenge-starter)
at commit 5b46798 (2026-09-23), MIT licensed. What is in it:

| Path | What |
|---|---|
| `fusion_scoring/` | The competition metric, byte-identical to what Codabench runs; `fusioneq.score` imports it |
| `data_fixes.py` | The two published data errata fixes (DIII-D Ip time axis, MAST Thomson channel 0) |
| `experiments.py`, `experiments_torch.py` | Reference baselines: PCA+Ridge, random forest, MLP, conv decoder |
| `submission_skeleton.py`, `validate_submission.py`, `push_predictions.py` | Build, validate and push a submission, write the Codabench pointer zip |
| `local_score.py` | Their held-out scorer (ours in `fusioneq.score` mirrors it on cached frames) |
| `fusion_data_provider.py` | dFL visualizer plugin; point the dFL desktop app at it to browse `parquet_data/` |
| `MODELING_GUIDE.md`, `README.md` | Their docs |

`parquet_data/` demo shots are Git LFS objects and are not copied here; they live on the A100 at
`~/fusion/starter/parquet_data/`. The full dataset (98 GB) is at `~/fusion/data/` there.
