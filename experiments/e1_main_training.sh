#!/usr/bin/env bash
# E1 — Main training & model comparison  (PathogenKG §3.1 / Table 4, adapted to PKT).
# For each task (A=DTI, B=TREATS) x each model (compgcn, rgcn): RUNS seeds, EPOCHS epochs,
# filtered evaluation. Produces per-run + aggregated AUROC/AUPRC/MRR/Hits@k and saves the
# trained model under models/<task>_<dataset>_<timestamp>/.
#
# Usage:   PROTOCOL=v2 bash experiments/e1_main_training.sh      # rgcn, compgcn, distmult baseline
#          RUNS=12 EPOCHS=400 MODELS="compgcn rgcn" bash experiments/e1_main_training.sh   # legacy v1
#          TASKS=A bash experiments/e1_main_training.sh          # only Task A (DTI)
#          CFG_A=PKT-DTI-best-v2b bash experiments/e1_main_training.sh   # force a tuned config
# Logs: experiments/logs/e1_<task>_<model>_<ts>.log (v1) or experiments/logs/v2/e1_... (v2).
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

TASKS="${TASKS:-A B}"          # which task graphs to train on
CFG_A="${CFG_A:-}"             # optional config override per task (e.g. a differently-suffixed HPO)
CFG_B="${CFG_B:-}"

run_one () {  # $1=tsv  $2=task  $3=model  $4=config override (may be empty)
  local tsv="$1" task="$2" model="$3" override="${4:-}"
  local cfg
  if [ -n "$override" ]; then cfg="$override"; else cfg="$(resolve_config "$task")"; fi
  # a config counts as v2-tuned when its name carries the v2 suffix (-v2, -v2b, ...)
  if [ "$PROTOCOL" = "v2" ] && [[ "$cfg" != "PKT-${task}-best-v2"* ]]; then
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

for t in $TASKS; do
  case "$t" in
    A|a) for m in $MODELS; do run_one "$TSV_A" "$TASK_A" "$m" "$CFG_A"; done ;;
    B|b) for m in $MODELS; do run_one "$TSV_B" "$TASK_B" "$m" "$CFG_B"; done ;;
    *) echo "[E1] unknown task '$t' (use A, B or \"A B\")" >&2; exit 1 ;;
  esac
done

echo "[E1] done. Trained models are in models/ ; logs in ${LOG_DIR}/"
