"""Rewrite the Results section of the W&B experiment-log report in place: official scores, a verdict per experiment
(docs/experiments/fusioneq/verdicts.yaml), the top local runs from results/*_score.json, and every run under a
collapsed Full run log.

uv run python scripts/update_wandb_report.py            # edit and save
uv run python scripts/update_wandb_report.py --dry-run  # print the new section only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import wandb_workspaces.reports.v2 as wr
import yaml
from autoresearch.wandb_report import replace_section, results_table, verdict_table
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
REPORT_URL = "https://wandb.ai/chaleong/fusion-equilibrium/reports/Fusion-Equilibrium-Challenge:-experiment-log--VmlldzoxODAxNjExMg=="
LOCAL_TOP = 15
FULL_LOG = "Full run log"
CONS = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
COLUMNS = ["Setup", "S", "R2psi", "R2 q95,betaN", "1 - D_LCFS", "Consistency", *CONS]
VERDICTS = ROOT / "docs" / "experiments" / "fusioneq" / "verdicts.yaml"
PR_URL = "https://github.com/charleneleong-ai/neurips-workshops-2026/pull"
GATES = {
    "G1": "beats the matched-seed baseline on the 40 held-out shots by more than seed noise (0.0013 for one model, "
    "about 0.0002 for an ensemble)",
    "G2": "lifts the blend on the held-out shots (from 8 Oct: wins most 40-shot groups of the 160 never-scored "
    "shots), then beats the official best on Codabench",
}


def load_scores(results: Path) -> list[dict[str, Any]]:
    return sorted((json.loads(p.read_text()) for p in results.glob("*_score.json")), key=lambda d: -d["S"])


def model_type(d: dict[str, Any]) -> str:
    """The score json's `model` field; older jsons predate it, so fall back to their distinguishing keys."""
    if "model" in d:
        return d["model"]
    for key, model in (("members", "ensemble"), ("levels", "agnostic"), ("hidden", "mlp")):
        if key in d:
            return model
    return "pca_ridge"


def describe(d: dict[str, Any]) -> str:
    match model_type(d):
        case "blend":
            return f"blend `{d['spec']}`"
        case "ensemble":
            return f"ensemble of {len(d['members'])}: {', '.join(d['members'])}"
        case "agnostic":
            m = d.get("mast_check") or {}
            mast = f"; MAST closed {m['closed_frac']:.0%}, R_geo {m['r_geo_median']:.2f} m" if m else ""
            return f"machine-agnostic U-Net ({d['levels']} levels, base {d['base']}), {d['epochs']} ep{mast}"
        case "mlp":
            ctx = f", context {d['context']}" if d.get("context") else ""
            vac = "" if d.get("use_vacuum", True) else ", no vacuum term"
            return f"MLP {d['hidden']}x{d['layers']} on PCA {d['n_components']}{ctx}{vac}, {d['epochs']} ep"
    vac = "on psi - psi_vac" if d.get("use_vacuum", True) else "on raw psi"
    return f"PCA {d['n_components']} + ridge {vac}"


def runs_table(scores: list[dict[str, Any]]) -> str:
    rows = [
        {
            "tag": d["tag"],
            "wandb_url": d.get("wandb_url"),
            "Setup": describe(d),
            "S": d["S"],
            "R2psi": max(0, d["r2_psi"]),
            "R2 q95,betaN": max(0, d["r2_qb"]),
            "1 - D_LCFS": 1 - min(1, d["dlcfs"]),
            "Consistency": d["consistency"],
            **{k: (d.get("r2_cons_each") or {}).get(k) or 0.0 for k in CONS},
        }
        for d in scores
    ]
    return results_table(rows, COLUMNS)


def verdicts_markdown() -> str:
    """One row per experiment: baseline, promotion gate, result, verdict and PR."""
    gates = "\n".join(f"- **{k}**: {v}." for k, v in GATES.items())
    entries = yaml.safe_load(VERDICTS.read_text())
    return (
        "### Experiment verdicts\n\nPROMOTE = shipped in a submission that set a new official best; KEEP = kept for "
        f"later work; DISCARD = closed. Promotion gates:\n\n{gates}\n\n" + verdict_table(entries, PR_URL) + "\n"
    )


def results_markdown(scores: list[dict[str, Any]]) -> str:
    best = scores[0]
    note = (
        f"\nTop {LOCAL_TOP} of {len(scores)} runs. Best so far: **{best['tag']}** at S = {best['S']:.4f}. "
        "Scores are the official composite on the same 40 held-out DIII-D shots (shot-level split, seed 0).\n"
    )
    return (
        "## Results\n\n"
        + leaderboard_markdown()
        + verdicts_markdown()
        + "### Local held-out shots\n\n"
        + runs_table(scores[:LOCAL_TOP])
        + note
    )


def leaderboard_markdown(path: Path = ROOT / "results" / "codabench.json") -> str:
    """Official public-test scores from Codabench submissions, newest first; empty until one lands."""
    if not path.exists():
        return ""
    subs = json.loads(path.read_text())
    lines = [
        "### Codabench public test (development phase)\n",
        "| Submitted | Model | Rank | DIII-D S | R2psi | R2 q95,betaN | 1 - D_LCFS | Consistency | MAST S | G_ratio |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(subs, key=lambda e: e["submitted"], reverse=True):
        c = e["scores"]
        if c is None:
            if not e["submitted"].startswith("held"):
                lines.append(f"| {e['submitted']} | {e['model']} | pending |" + " |" * 7)
            continue
        cells = [
            e["submitted"],
            e["model"],
            f"{e['rank'] or '-'}/{e['entries']}",
            c["d3d_S"],
            c["d3d_r2_psi"],
            c["d3d_r2_qb"],
            1 - c["d3d_dlcfs"],
            c["d3d_consistency"],
            c["mast_S"],
            c["g_ratio"],
        ]
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in cells) + " |")
    return "\n".join(lines) + "\n\n" + mast_physics_markdown()


def mast_physics_markdown(path: Path = ROOT / "results" / "mast_physics.json") -> str:
    """Zero-shot MAST flux: its DIII-D calibration check and the LCFS closure rate on MAST predictions."""
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    c, m = d["calibration"], d["mast_lcfs"]
    rows = "\n".join(
        f"| {r['variant']} | {r['S']:.3f} | {r['r2_psi']:.2f} | {r['lcfs']:.2f} | {r['cons']:.2f} |"
        for r in d["diii_d_check"]
    )
    return (
        "### Zero-shot MAST physics flux\n\n"
        "MAST gets the coil vacuum flux plus the measured plasma current spread over a Thomson-located ellipse. "
        f"Fitted on DIII-D: scale {c['scale']:.3f}, plasma/vacuum {c['plasma_ratio']:.3f}, offset {c['offset']:.3f}, "
        f"and {c['ecoil_turns']:.0f} missing E-coil turns (DIII-D only; MAST ships its solenoid turn by turn).\n\n"
        "| Variant (40 held-out DIII-D shots, EFIT-boundary geometry) | S | R2psi | 1 - D_LCFS | Consistency |\n"
        "|---|---|---|---|---|\n" + rows + "\n\n"
        f"On MAST the prediction closes the LCFS in {m['closed']}/{m['frames']} sampled frames "
        f"(median R_geo {m['r_geo_median']:.2f} m, elongation {m['kappa_median']:.2f}).\n\n"
        + scalars_line(d.get("scalars"))
    )


def scalars_line(sc: dict[str, Any] | None) -> str:
    if not sc:
        return ""
    return (
        "q95 and betaN come from textbook formulas (TF field, Ip, the predicted boundary, Thomson pressure) with one "
        f"constant each fitted on DIII-D: held-out DIII-D R2 q95 {sc['diii_d_r2_q95']:.3f}, betaN "
        f"{sc['diii_d_r2_betaN']:.3f}. On MAST the medians are q95 {sc['mast_q95_p10_50_90'][1]:.2f}, "
        f"betaN {sc['mast_betaN_p10_50_90'][1]:.2f}.\n\n"
    )


def panels() -> list[Any]:
    return [
        wr.LinePlot(title="holdout R2psi during training", x="Step", y=["holdout_r2_psi"], layout=wr.Layout(w=12, h=8)),
        wr.LinePlot(title="train loss", x="Step", y=["train_loss"], log_y=True, layout=wr.Layout(x=12, w=12, h=8)),
        wr.BarPlot(title="official composite S", metrics=["S"], layout=wr.Layout(y=8, w=12, h=8)),
        wr.BarPlot(title="consistency", metrics=["consistency"], layout=wr.Layout(x=12, y=8, w=12, h=8)),
        wr.MediaBrowser(media_keys=["figures/flux"], num_columns=2, layout=wr.Layout(y=16, w=24, h=14)),
        wr.MediaBrowser(media_keys=["figures/scalars"], num_columns=2, layout=wr.Layout(y=30, w=24, h=10)),
    ]


def main(results: Path = ROOT / "results", dry_run: bool = False) -> None:
    scores = load_scores(results)
    section = results_markdown(scores)
    if dry_run:
        print(section)
        return
    full = f"All {len(scores)} runs on the 40 held-out shots, best first.\n\n" + runs_table(scores)
    url = replace_section(REPORT_URL, "Results", section, panels(), collapsed={FULL_LOG: full})
    logger.info(f"report updated with {len(scores)} runs: {url}")


if __name__ == "__main__":
    typer.run(main)
