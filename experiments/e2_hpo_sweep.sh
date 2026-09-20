#!/usr/bin/env bash
# E2 — Hyperparameter optimisation  (PathogenKG §3.2/§3.3 — Bayesian W&B sweep).
# Optimises the composite metric M = 0.2*AUROC + 0.4*AUPRC + 0.4*MRR on the validation set.
# Runs one sweep per task; models swept = AVAILABLE_MODELS in tuning_hyperparameter.py
# (v2: rgcn + compgcn + distmult baseline). Best configs then go into src/models_params.json
# (PKT-<TASK>-best-v2 for protocol v2) for the final E1 runs.
#
# PREREQUISITES:
#   pip install wandb && wandb login
#   (W&B entity/project are hardcoded in tuning_hyperparameter.py = RelationalPKT;
#    override here with WANDB_ENTITY / WANDB_PROJECT env vars if needed.)
#
# Usage:   bash experiments/e2_hpo_sweep.sh A     # sweep Task A (project RelationalPKT-DTI)
#          bash experiments/e2_hpo_sweep.sh B     # sweep Task B (project RelationalPKT-TREATS)
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

WHICH="${1:-A}"

# Training protocol of the sweep (tuning_hyperparameter.py): v2 = consolidated (default), v1 = legacy.
# v2 sweeps log to separate W&B projects (suffix -v2) so they never mix with the v1 trials.
export PKT_HPO_PROTOCOL="${PKT_HPO_PROTOCOL:-v2}"
if [ "$PKT_HPO_PROTOCOL" = "v2" ]; then HPO_SUFFIX="${HPO_SUFFIX:--v2}"; else HPO_SUFFIX="${HPO_SUFFIX:-}"; fi
export HPO_SUFFIX

if [ "$WHICH" = "A" ]; then
  export PKT_TSV="$TSV_A"; export PKT_TASK="$TASK_A"; export WANDB_PROJECT="RelationalPKT-DTI${HPO_SUFFIX}"
else
  export PKT_TSV="$TSV_B"; export PKT_TASK="$TASK_B"; export WANDB_PROJECT="RelationalPKT-TREATS${HPO_SUFFIX}"
fi
if [ "$PKT_HPO_PROTOCOL" = "v2" ]; then   # v2 GNNs converge later (E0: best epoch 292/300)
  export PKT_HPO_EPOCHS="${PKT_HPO_EPOCHS:-500}"
  export PKT_HPO_PATIENCE="${PKT_HPO_PATIENCE:-10}"   # evaluations (every 5 epochs)
else
  export PKT_HPO_EPOCHS="${PKT_HPO_EPOCHS:-200}"
  export PKT_HPO_PATIENCE="${PKT_HPO_PATIENCE:-50}"
fi
export PKT_HPO_RUNS="${PKT_HPO_RUNS:-100}"

log="${LOG_DIR}/e2_hpo_${WHICH}_$(date +%Y%m%d_%H%M%S).log"
echo "[E2] sweep task=$WHICH tsv=$PKT_TSV project=$WANDB_PROJECT runs=$PKT_HPO_RUNS -> $log"
run_logged "$log" python tuning_hyperparameter.py
echo "[E2] done. Inspect the sweep on W&B; copy the best config into src/models_params.json."
