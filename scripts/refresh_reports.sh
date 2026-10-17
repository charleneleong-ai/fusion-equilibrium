#!/usr/bin/env bash
# Pull results from the A100 and refresh every report: W&B (in place), LaTeX table + PDF, and the dashboard HTML.
# The dashboard is then republished as the Claude Artifact (https://claude.ai/artifact/ED9YgJrHSqJUzuBrJgowGw)
# by republishing report/dashboard.html from the session that owns it.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az 'pi-a100-80gb:~/fusion/results/*_score.json' results/
rsync -az 'pi-a100-80gb:~/fusion/results/figures/' results/figures/
rsync -az 'pi-a100-80gb:~/fusion/repo/experiments/fusioneq/' experiments/fusioneq/
uv run python scripts/update_wandb_report.py
uv run python scripts/report_table.py >/dev/null
(cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true)
uv run python scripts/build_dashboard.py
