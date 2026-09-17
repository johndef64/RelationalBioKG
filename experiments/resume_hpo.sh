#!/usr/bin/env bash
# Resume an interrupted HPO sweep — re-attaches an agent to the SAVED sweep id
# (experiments/hpo_sweeps/<project>.txt) instead of starting a new sweep. The Bayesian
# search continues, reusing the trials already logged on W&B.
#
# PKT_HPO_RUNS is the TOTAL number of trials per model (default 30, as e2_hpo_tandem.sh):
# the finished trials are counted on W&B and only the missing ones are run; a model whose
# sweep is already complete is skipped, a model whose sweep never started gets a new sweep.
#
# ONLY for resuming a sweep that was interrupted. Do NOT use it to add trials on a CHANGED search
# space: the grid is frozen inside the W&B sweep when the sweep is created, so re-attaching would
# keep searching the OLD grid, and the finished-trial count would make it skip the model outright
# (30 done >= the target you asked for -> 0 runs, no training, no warning). To search a widened
# grid, launch a NEW sweep in the same project instead -- that is what e2_hpo_sweep.sh does without
# PKT_HPO_RESUME. The old trials stay in the project and get_best_hpo_config.py ranks them all
# together, so nothing is lost and nothing needs deleting.
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
