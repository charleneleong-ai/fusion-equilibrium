"""Schedule planner and score-json extractor: tags, resumability, status labelling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from autoresearch.results import log_experiment

from experiments.autoresearch import Schedule, SchedulePlanner, ScoreJsonExtractor


@pytest.fixture
def schedule(tmp_path: Path) -> Schedule:
    p = tmp_path / "v9.yaml"
    p.write_text(
        "common_overrides: [--epochs, 1]\niters:\n"
        "  - {description: a, overrides: [--context, 0]}\n"
        "  - {description: b, overrides: [--context, 2]}\n"
        "  - {description: c, command: ensemble, overrides: [--members, 'x,y']}\n"
    )
    return Schedule.load(p)


def test_planner_builds_commands_and_skips_logged_iters(schedule, tmp_path):
    planner = SchedulePlanner(schedule, "mlp", "python", tmp_path)
    plans = list(planner.plan_iters([{"description": "v9_i00 a"}]))
    assert [p.description for p in plans] == ["v9_i01 b", "v9_i02 c"]
    assert plans[0].cmd[-4:] == ["--context", "2", "--tag", "v9_i01"] and "--epochs" in plans[0].cmd
    # an iter with its own command runs it without the schedule command's common overrides
    assert plans[1].cmd[3:] == ["ensemble", "--members", "x,y", "--tag", "v9_i02"]


def test_extractor_labels_against_full_history(schedule, tmp_path):
    exp = tmp_path / "experiments"
    p0, p1, _ = list(SchedulePlanner(schedule, "mlp", "python", tmp_path).plan_iters([]))
    ex = ScoreJsonExtractor(tmp_path, "mlp", exp)
    (tmp_path / "v9_i00_score.json").write_text(json.dumps({"S": 0.80, "r2_psi": 0.9, "epochs": 1}))
    (tmp_path / "v9_i01_score.json").write_text(json.dumps({"S": 0.90, "epochs": 1}))
    assert ex.extract(p0, None, 0)[0]["status"] == "BASELINE"
    # a better row logged by an earlier sweep makes this sweep's 0.90 a DISCARD, not a KEEP
    log_experiment(experiments_dir=exp, tag="fusioneq", config_name="mlp", score=0.98, status="KEEP")
    assert ex.extract(p1, None, 0)[0]["status"] == "DISCARD"
    assert ex.extract(p1, None, 1)[0]["status"] == "CRASH"
