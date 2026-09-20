#!/usr/bin/env bash
# =============================================================================
# CONVERGENCE CHECK — are the HPO winners limited by the search budget?
# =============================================================================
# The v2 HPO stopped almost every trial at the 500-epoch ceiling (DTI: 27/30 R-GCN, 23/30 DistMult,
# 18/26 CompGCN), and the GNNs' best learning rate (0.01) is the TOP of their grid, while DistMult's
# best (0.03) is interior. So "DistMult wins" could still be an artefact of the search bounds.
#
# This script re-runs the tuned config of each model with a much larger epoch budget and, for the
# GNNs, also with the learning rates that were outside their grid. One run each: it is a check on
# the budget, not a comparison between models (that is E1, with 12 seeds).
#
# Protocol v2 throughout (fixed split, val-M checkpoint, disjoint supervision, warm metrics).
#
# USAGE (after the HPO has written PKT-<TASK>-best-v2):
#   bash experiments/convergence_check.sh                     # DTI, 3 models
#   CONV_TASKS="DTI TREATS" bash experiments/convergence_check.sh
#   CONV_EPOCHS=3000 CONV_MODELS=rgcn bash experiments/convergence_check.sh
#   python experiments/convergence_summary.py                 # table (also run at the end)
#
# READING THE TABLE (experiments/logs/v2/conv/convergence_summary.md):
#   * "best ep." still at the ceiling  -> the budget is STILL the limit: re-run with CONV_EPOCHS=3000
#   * a GNN improving a lot at lr 0.03/0.1 -> its HPO grid was cut too low: widen it and re-tune
#   * GNNs converged, still below DistMult on MRR -> the result is solid, write it up
#
# COST: tasks x (2 GNNs x 3 lr + 1 DistMult) = 7 runs per task, each up to CONV_EPOCHS epochs.
# Crash-resume: a completed (task, model, tag) is skipped.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PROTOCOL=v2
source experiments/config.sh

CONV_TASKS="${CONV_TASKS:-DTI}"
CONV_MODELS="${CONV_MODELS:-rgcn compgcn distmult}"
CONV_EPOCHS="${CONV_EPOCHS:-1500}"
CONV_RUNS="${CONV_RUNS:-1}"
CONV_GNN_LRS="${CONV_GNN_LRS:-0.03 0.1}"    # extra learning rates, beyond the tuned one
CONV_DM_LRS="${CONV_DM_LRS:-}"              # DistMult's optimum is interior: tuned lr only
CONV_RESUME="${CONV_RESUME:-1}"
CONV_DIR="${LOG_DIR}/conv"
mkdir -p "$CONV_DIR"

tsv_of () { [ "$1" = "$TASK_A" ] && echo "$TSV_A" || echo "$TSV_B"; }

done_already () {   # $1 = log glob prefix, $2 = number of runs
  [ "$CONV_RESUME" = "1" ] && grep -lq "Completed run $(( $2 - 1 ))/$2" "$1"_*.log 2>/dev/null
}

run_one () {   # $1 task  $2 model  $3 tag  $4 config  $5.. extra flags
  local task="$1" model="$2" tag="$3" cfg="$4"; shift 4
  local prefix="${CONV_DIR}/conv_${task}_${model}_${tag}"
  if done_already "$prefix" "$CONV_RUNS"; then
    echo "[conv] skip ${task}/${model}/${tag} (already complete)"; return 0
  fi
  local log="${prefix}_$(date +%Y%m%d_%H%M%S).log"
  echo "[conv] ${task} | ${model} | ${tag} | config=${cfg} epochs=${CONV_EPOCHS} -> ${log}"
  # shellcheck disable=SC2068
  run_logged "$log" python train_and_eval.py --tsv "$(tsv_of "$task")" --task "$task" --model "$model" --config "$cfg" \
    --runs "$CONV_RUNS" --epochs "$CONV_EPOCHS" $COMMON_FLAGS $@ \
    || echo "[conv] FAILED ${task}/${model}/${tag} (see ${log})"
}

echo "[conv] tasks='$CONV_TASKS' models='$CONV_MODELS' epochs=$CONV_EPOCHS runs=$CONV_RUNS"
for task in $CONV_TASKS; do
  cfg="$(resolve_config "$task")"
  case "$cfg" in
    *-v2) ;;
    *) echo "[conv] WARNING: config '$cfg' was not tuned under v2 — run the HPO extraction first" ;;
  esac

  for model in $CONV_MODELS; do
    # 1) the tuned config as-is, only with a larger epoch budget
    run_one "$task" "$model" "tuned" "$cfg"
    # 2) the learning rates that were outside (GNNs) or beyond (DistMult) the HPO grid
    lrs="$CONV_GNN_LRS"; [ "$model" = "distmult" ] && lrs="$CONV_DM_LRS"
    for lr in $lrs; do
      run_one "$task" "$model" "lr${lr}" "$cfg" --learning_rate "$lr"
    done
  done
done

python experiments/convergence_summary.py --logdir "$CONV_DIR" --epochs_cap "$CONV_EPOCHS" \
  || echo "[conv] summary failed — run: python experiments/convergence_summary.py"
echo "[conv] done. Table: ${CONV_DIR}/convergence_summary.md"
