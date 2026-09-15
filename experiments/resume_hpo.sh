#!/usr/bin/env bash
# Resume an interrupted HPO sweep — re-attaches an agent to the SAVED sweep id
# (experiments/hpo_sweeps/<project>.txt) instead of starting a new sweep. The Bayesian
# search continues, reusing the trials already logged on W&B.
#
# PKT_HPO_RUNS is the TOTAL number of trials per model (default 30, as e2_hpo_tandem.sh):
# the finished trials are counted on W&B and only the missing ones are run; a model whose
# sweep is already complete is skipped, a model whose sweep never started gets a new sweep.
#
# Usage:  bash experiments/resume_hpo.sh A        # resume Task A (DTI) sweeps
#         bash experiments/resume_hpo.sh B        # resume Task B (TREATS) sweeps
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-A}"
export PKT_HPO_RESUME=1
export PKT_HPO_RUNS="${PKT_HPO_RUNS:-30}"
echo "[resume] task $WHICH: completing sweeps up to PKT_HPO_RUNS=$PKT_HPO_RUNS trials per model"
bash experiments/e2_hpo_sweep.sh "$WHICH"
