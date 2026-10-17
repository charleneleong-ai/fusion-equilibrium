"""Render report/dashboard.html (the Claude Artifact) from results/*_score.json, results rows, verdicts and figures.

uv run python scripts/build_dashboard.py
"""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

import typer
import yaml
from update_wandb_report import CONS, PR_URL, model_type
from update_wandb_report import describe as describe_score

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments" / "fusioneq"
POINTER_SHA = re.compile(r"_([0-9a-f]{8})\.zip$")
# composite weights of the DIII-D score: S = .55 R2psi + .15 R2(q95, betaN) + .10 (1 - D_LCFS) + .20 consistency
TERM_WEIGHTS = {"r2_qb": 0.15, "dlcfs": 0.10, "consistency": 0.20}
UNMOVED_TOL = 1e-4  # an official term within this of its current value counts as "not moved"
G2_MIN_WINS = 3  # "wins most of the five 40-shot groups"
ORDER = ["pca_ridge", "mlp_", "v1_", "v2_", "v3_", "v4_", "v5_", "v6_", "v7_", "v8_", "agnostic", "blend"]


def sweep_of(d: dict[str, Any]) -> str:
    """Schedule-run iters are tagged <sweep>_i<NN>; standalone runs group by model type (agnostic) or as pre-sweep."""
    tag = d["tag"]
    if (kind := model_type(d)) in ("agnostic", "blend"):
        return kind
    return (
        tag.rsplit("_i", 1)[0]
        if "_i" in tag and tag[-1].isdigit() and tag.split("_")[0].startswith("v")
        else "pre-sweep"
    )


def order_key(tag: str) -> tuple[int, str]:
    return next((i for i, p in enumerate(ORDER) if tag.startswith(p)), len(ORDER)), tag


def describe(d: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    """The sweep's own iter description when there is one, else the W&B report's description of the score json."""
    if d["tag"] in rows:
        return rows[d["tag"]]["description"].split(" ", 1)[-1]
    return describe_score(d)


def load_runs() -> list[dict[str, Any]]:
    rows = {}
    for f in (ROOT / "experiments" / "fusioneq").glob("*/results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["description"].split(" ")[0]] = r
    runs = []
    for p in (ROOT / "results").glob("*_score.json"):
        d = json.loads(p.read_text())
        runs.append(
            {
                "tag": d["tag"],
                "sweep": sweep_of(d),
                "desc": describe(d, rows),
                "S": d["S"],
                "r2_psi": d["r2_psi"],
                "r2_qb": d["r2_qb"],
                "lcfs": 1 - min(1, d["dlcfs"]),
                "cons": d["consistency"],
                "cons_each": {k: (d.get("r2_cons_each") or {}).get(k) for k in CONS},
                "epochs": d.get("epochs"),
                "amp": bool(d.get("amp")),
                "min": round(d.get("runtime_min", 0) or 0, 1),
                "status": rows.get(d["tag"], {}).get("status", ""),
                "url": d.get("wandb_url", ""),
                "spec": d.get("spec"),
            }
        )
    return sorted(runs, key=lambda r: order_key(r["tag"]))


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def grid_model(best: dict[str, Any]) -> str:
    """The newest grid model in the best blend's spec, whose shape figure the dashboard shows."""
    if not best.get("spec"):
        return ""
    models = (yaml.safe_load((ROOT / best["spec"]).read_text()).get("grid") or {}).get("models") or [""]
    return models[-1]


def official_entries(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scored Codabench entries in submission order, each flagged when it raised the official DIII-D best."""
    entries, best = [], -1.0
    for e in leaderboard:
        if not (c := e.get("scores")):
            continue
        sha = m.group(1) if (m := POINTER_SHA.search(e.get("pointer") or "")) else ""
        entries.append(
            {
                "date": e["submitted"][:10],
                "sha": sha,
                "rank": e.get("rank"),
                "entries": e.get("entries"),
                "S": c["d3d_S"],
                "r2_qb": c["d3d_r2_qb"],
                "dlcfs": c["d3d_dlcfs"],
                "consistency": c["d3d_consistency"],
                "g_ratio": c.get("g_ratio"),
                "mast_S": c.get("mast_S"),
                "new_best": c["d3d_S"] > best,
            }
        )
        best = max(best, c["d3d_S"])
    return entries


def unmoved_since(entries: list[dict[str, Any]], key: str) -> str:
    """Date of the earliest entry from which every later official `key` stays within UNMOVED_TOL of the latest."""
    now, since = entries[-1][key], entries[-1]["date"]
    for e in reversed(entries):
        if abs(e[key] - now) > UNMOVED_TOL:
            break
        since = e["date"]
    return since


def term_gaps(best: dict[str, Any], leaders: dict[str, float]) -> list[dict[str, Any]]:
    """Each term's gap to the leaders, weighted by its share of S (D_LCFS counts as 1 - D_LCFS), largest first."""
    gaps = [
        {
            "term": k,
            "ours": best[k],
            "leaders": leaders[k],
            "weighted": w * (best[k] - leaders[k] if k == "dlcfs" else leaders[k] - best[k]),
        }
        for k, w in TERM_WEIGHTS.items()
    ]
    return sorted(gaps, key=lambda g: -g["weighted"])


def best_mast(entries: list[dict[str, Any]], doc: dict[str, Any]) -> dict[str, Any]:
    """Best label-free MAST S overall and among rule-3-compliant (mask-free scalar) entries; ties go to the earliest."""
    start = next(i for i, e in enumerate(entries) if e["sha"] == doc["label_free_from"])
    label_free = [e for e in entries[start:] if e["mast_S"] is not None]
    compliant = [e for e in label_free if e["sha"] in set(doc.get("compliant_mast") or [])]
    return {
        "best_mast": max(label_free, key=lambda e: e["mast_S"]),
        "best_compliant_mast": max(compliant, key=lambda e: e["mast_S"], default=None),
    }


def headline(entries: list[dict[str, Any]], doc: dict[str, Any]) -> dict[str, Any]:
    leaders = doc["leaders"]
    best = max(entries, key=lambda e: e["S"])
    return {
        "best": best,
        **best_mast(entries, doc),
        "baseline": entries[0],
        "leader_S": leaders["S"],
        "qb_since": unmoved_since(entries, "r2_qb"),
        "gaps": term_gaps(best, leaders),
    }


def sign_test_p(wins: int, n: int) -> float:
    """One-sided sign test: P(at least `wins` positive of `n` paired groups) under a fair coin."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2**n


def screen_row(s: dict[str, Any], official_bests: set[str]) -> dict[str, Any]:
    groups = s.get("groups")
    d160 = None if s["s160"] is None or s["baseline_160"] is None else round((s["s160"] - s["baseline_160"]) * 1e6)
    wins = sum(g > 0 for g in groups) if groups else None
    passed = bool(groups) and wins >= G2_MIN_WINS and (d160 is None or d160 > 0)
    return s | {
        "d160": d160,
        "wins": wins,
        "discordant": len(groups) - wins if groups else None,
        "p": sign_test_p(wins, len(groups)) if groups else None,
        "passed": passed,
        "new_best": s.get("commit") in official_bests,
    }


def funnel(runs: list[dict[str, Any]], screens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [s for s in screens if s["passed"]]
    submitted = [s for s in passed if s.get("commit")]
    return [
        {"stage": "Models trained, scored on 40 held-out shots", "n": sum(r["sweep"] != "blend" for r in runs)},
        {"stage": "Blend candidates screened on 160 shots (from 8 Oct)", "n": len(screens)},
        {"stage": f"Passed G2 (160-shot gain, wins >= {G2_MIN_WINS} of 5 groups)", "n": len(passed)},
        {"stage": "Submitted to Codabench", "n": len(submitted)},
        {"stage": "Set a new official best", "n": sum(s["new_best"] for s in submitted)},
    ]


def ledger(screens_doc: dict[str, Any]) -> dict[str, Any]:
    verdicts = yaml.safe_load((EXP / "verdicts.yaml").read_text())
    running = [r | {"verdict": "RUNNING"} for r in screens_doc.get("running") or []]
    return {"rows": verdicts + running, "not_established": screens_doc.get("not_established") or []}


def evidence(runs: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    """The research-artifact charts: headline strip, scoreboards, per-group gains, funnel and verdict ledger."""
    doc = yaml.safe_load((EXP / "screens.yaml").read_text())
    entries = official_entries(leaderboard)
    bests = {e["sha"] for e in entries if e["new_best"]}
    screens = [screen_row(s, bests) for s in doc["screens"]]
    return {
        "built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "official": entries,
        "headline": headline(entries, doc),
        "group_names": doc["group_names"],
        "screens": screens,
        "funnel": funnel(runs, screens),
        "ledger": ledger(doc),
    }


def main(out: Path = ROOT / "report" / "dashboard.html") -> None:
    runs = load_runs()
    best = max(runs, key=lambda r: r["S"])
    # ensembles have no figures of their own: show the best run that does
    figs = ROOT / "results" / "figures"
    shown = max((r for r in runs if (figs / f"{r['tag']}_flux.png").exists()), key=lambda r: r["S"], default=best)
    grid = grid_model(best)
    template = (ROOT / "scripts" / "dashboard_template.html").read_text()
    lb_path = ROOT / "results" / "codabench.json"
    leaderboard = json.loads(lb_path.read_text()) if lb_path.exists() else []
    mast_path = ROOT / "results" / "mast_physics.json"
    mast = json.loads(mast_path.read_text()) if mast_path.exists() else None
    payload = json.dumps(
        {
            "runs": runs,
            "best": best["tag"],
            "gate": 0.85,
            "leaderboard": leaderboard,
            "mast": mast,
            "pr_url": PR_URL,
            **evidence(runs, leaderboard),
        }
    )
    out.write_text(
        template.replace("__DATA__", payload)
        .replace("__FIGURE__", png_uri(figs / f"{shown['tag']}_flux.png"))
        .replace("__BEST__", shown["tag"])
        .replace("__GRID_FIGURE__", png_uri(figs / f"{grid}_shape.png"))
        .replace("__GRID__", grid)
    )
    typer.echo(f"{out} ({out.stat().st_size / 1e3:.0f} kB): {len(runs)} runs, best {best['tag']} S={best['S']:.4f}")


if __name__ == "__main__":
    typer.run(main)
