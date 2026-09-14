#!/usr/bin/env bash
# E1 — Main training & model comparison  (PathogenKG §3.1 / Table 4, adapted to PKT).
# For each task (A=DTI, B=TREATS) x each model (compgcn, rgcn): RUNS seeds, EPOCHS epochs,
# filtered evaluation. Produces per-run + aggregated AUROC/AUPRC/MRR/Hits@k and saves the
# trained model under models/<task>_<dataset>_<timestamp>/.
#
# Usage:   PROTOCOL=v2 bash experiments/e1_main_training.sh      # rgcn, compgcn, distmult baseline
#          RUNS=12 EPOCHS=400 MODELS="compgcn rgcn" bash experiments/e1_main_training.sh   # legacy v1
# Logs: experiments/logs/e1_<task>_<model>_<ts>.log (v1) or experiments/logs/v2/e1_... (v2).
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

run_one () {  # $1=tsv  $2=task  $3=model
  local tsv="$1" task="$2" model="$3"
  local cfg; cfg="$(resolve_config "$task")"   # v2: PKT-<task>-best-v2 ; v1: PKT-<task>-best ; else $HP_CONFIG
  if [ "$PROTOCOL" = "v2" ] && [ "$cfg" != "PKT-${task}-best-v2" ]; then
    echo "[E1] WARNING: PROTOCOL=v2 but config '$cfg' was not tuned under v2 (run the v2 HPO first)."
    if [ "$model" = "distmult" ]; then
      echo "[E1] skipping distmult: without a v2 config it would reuse the R-GCN learning rate (unfair baseline)."
      return 0
    fi
  fi
  local log="${LOG_DIR}/e1_${task}_${model}_$(date +%Y%m%d_%H%M%S).log"
  echo "[E1] task=$task model=$model config=$cfg -> $log"
  python train_and_eval.py \
    --tsv "$tsv" --task "$task" --model "$model" --config "$cfg" \
    --runs "$RUNS" --epochs "$EPOCHS" $COMMON_FLAGS 2>&1 | tee "$log"
}

for m in $MODELS; do run_one "$TSV_A" "$TASK_A" "$m"; done   # Task A (DTI)
for m in $MODELS; do run_one "$TSV_B" "$TASK_B" "$m"; done   # Task B (TREATS)

echo "[E1] done. Trained models are in models/ ; logs in ${LOG_DIR}/"
