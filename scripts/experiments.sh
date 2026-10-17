#!/usr/bin/env bash
# First pass on the A100: sklearn baseline, then MLP variants; one line per run in results/experiments.log.
set -u
cd ~/fusion/repo
P=../starter/.venv/bin/python
LOG=../results/experiments.log; mkdir -p ../results
run() { echo "[$(date -u +%T)] $*" >> "$LOG"; $P -m fusioneq.cli "$@" 2>&1 | grep -E "S=|R2psi|epoch" | tail -3 >> "$LOG"; }
run baseline --tag pca_ridge_vac --use-vacuum --n-components 64
run baseline --tag pca_ridge_raw --no-use-vacuum --n-components 64
run train --tag mlp_c0 --epochs 15 --context 0
run train --tag mlp_c2 --epochs 15 --context 2
run train --tag mlp_c2_wide --epochs 25 --context 2 --hidden 2048 --layers 6 --n-components 192
echo "[$(date -u +%T)] EXPERIMENTS DONE" >> "$LOG"
