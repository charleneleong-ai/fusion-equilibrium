"""The Final's training entrypoint: committed member recipes turned into an ordered rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusioneq import rebuild
from fusioneq.finish import FinishSpec
from fusioneq.grid import GridConfig
from fusioneq.model import NetConfig
from fusioneq.rebuild import MAST_PSI, MAST_SCAL, MEMBERS_DIR, Job, Rebuild, mast
from fusioneq.settings import load_config

FINAL = Path(__file__).parents[1] / "configs" / "blends" / "v21_16_grids3_fno2.yaml"
CONFIGS = {"mlp": NetConfig, "grid": GridConfig}


def one_second(_job: Job) -> float:
    return 1.0


@pytest.mark.parametrize("recipe", sorted(MEMBERS_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_committed_recipe_loads_into_its_config(recipe):
    load_config(CONFIGS[json.loads(recipe.read_text())["kind"]], recipe)  # an unknown or misspelt field raises


class TestRebuild:
    def test_final_spec_rebuilds_every_member_in_dependency_order(self, tmp_path):
        jobs = Rebuild(FINAL).jobs(["diii_d_public_test"], tmp_path)
        commands = [j.args[0] for j in jobs]
        assert commands.count("train") == 16 and commands.count("train-grid") == 3 and commands.count("fit-trees") == 2
        assert max(i for i, c in enumerate(commands) if c == "train") < commands.index("fit-trees")
        assert commands[0] == "prepare" and commands[-3:] == ["blend", "prepare", "build"]
        assert all(j.args[j.args.index("--stride") + 1] == "1" for j in jobs if j.args[0] == "fit-trees")

    def test_mast_config_adds_its_calibrations_flux_and_scalars_before_the_build(self, tmp_path):
        commands = [j.args[0] for j in Rebuild(FINAL).jobs(["diii_d_public_test", "mast_public_test"], tmp_path)]
        assert (
            commands.index("shift-fit")
            < commands.index("mast-physics")
            < commands.index("mast-shape")
            < len(commands) - 1
        )
        build = Rebuild(FINAL).jobs(["mast_public_test"], tmp_path)[-1].args
        assert build[build.index("--mast-psi") + 1] == MAST_PSI and build[build.index("--mast-scal") + 1] == MAST_SCAL

    def test_mast_path_drops_the_demo_shot_sweep_picks(self, tmp_path):
        args = [a for j in mast("mast_public_test") for a in j.args]
        assert [args[i + 1] for i, a in enumerate(args) if a in ("--alpha", "--frac")] == ["1.0", "0.25", "1.0"]

    def test_finishing_trains_its_targets_and_heads_after_the_blend_and_the_build_runs_it(self, tmp_path):
        recipe = tmp_path / "axis_li.json"
        FinishSpec({}, 1, (0, 1), {"li": 0.3}).save(recipe)
        jobs = Rebuild(FINAL, finish=recipe).jobs(["diii_d_public_test"], tmp_path)
        commands = [j.args[0] for j in jobs]
        assert commands.index("blend") < commands.index("finish-targets") < commands.index("train-heads")
        assert commands.count("train-heads") == 2 and jobs[-1].args[-2:] == ("--finish", str(recipe))
        assert "--finish" not in Rebuild(FINAL).jobs(["diii_d_public_test"], tmp_path)[-1].args

    def test_missing_recipe_names_the_file_to_add(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="v5_loss_mix_i03"):
            Rebuild(FINAL, members=tmp_path).train("v5_loss_mix_i03")

    def test_run_skips_finished_steps_always_builds_and_logs_each_step(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rebuild, "RESULTS", tmp_path)
        monkeypatch.setattr(rebuild, "CACHE_ROOT", tmp_path)
        monkeypatch.setattr(Job, "run", one_second)
        (tmp_path / "diii_d_train").mkdir()
        (tmp_path / "v5_loss_mix_i03").mkdir()
        r = Rebuild(FINAL)
        r.run(["diii_d_public_test"], tmp_path / "out")
        rows = [json.loads(line) for line in (tmp_path / f"rebuild_{FINAL.stem}.jsonl").read_text().splitlines()]
        assert len(rows) == len(r.jobs(["diii_d_public_test"], tmp_path / "out")) - 2
        assert rows[-1]["args"][0] == "build" and all(row["seconds"] == 1 for row in rows)
