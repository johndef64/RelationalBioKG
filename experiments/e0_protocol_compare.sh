#!/usr/bin/env bash
# =============================================================================
# E0 — PROTOCOL COMPARISON: legacy system (v1) vs consolidated system (v2)
# =============================================================================
# Run this BEFORE redoing HPO / ablation / E1. It answers one question: does the consolidated
# protocol (docs/piano_consolidamento_v2.md) change the results, and which change does what?
#
# Same code, same tuned configs (PKT-<TASK>-best, tuned under v1 -> the comparison is
# conservative for v2), same epochs budget. Only train_and_eval.py protocol flags change.
#
# VARIANTS (cumulative ladder; each adds ONE change to the previous one):
#   v1            legacy: x5 oversampling, 50% random undersampling, split changes per run,
#                 checkpoint + early stopping on val LOSS, train target edges in the graph
#   v1_fixsplit   + fixed split (seed 42)            -> evaluation change only, same model
#   s1_selectM    + checkpoint/early stopping on val M (patience PATIENCE_V2)
#   s2_negatives  + no oversampling, 5 explicit training negatives per positive
#   s3_fullgraph  + no background undersampling (full context graph)
#   v2            + disjoint supervision edges (ratio V2_DISJOINT)  == FLAGS_V2
# All variants also report cold-start-excluded test metrics (--warm_eval, reporting only).
#
# BASELINES (CMP_BASELINES=1): popularity (node degree, no learning) and DistMult without
# message passing (embeddings only, v2 protocol) on the fixed split. Full-batch training does ONE
# optimiser step per epoch, so DistMult is run on a small learning-rate grid (CMP_DISTMULT_LRS):
# with the GNN learning rate (1e-3) it barely trains and would be an unfairly weak baseline.
#
# MODES:  quick  -> v2 only, plus the baselines    (sanity check of a NEW target relation before
#                                                   spending a day on the HPO: is the model above
#                                                   the popularity floor, and where is DistMult?)
#         pair   -> v1, v2, v1_fixsplit            (minimum: old vs new protocol)
#         ladder -> all six variants (default)     (attributes the effect of each change)
#
# CMP_CONFIG overrides the tuned config used for every variant (default: the v1-tuned one).
#
# USAGE:
#   bash experiments/e0_protocol_compare.sh                    # ladder, both tasks, rgcn+compgcn
#   bash experiments/e0_protocol_compare.sh pair
#   TASKS=DTI CMP_CONFIG=PKT-DTI-best-v2 CMP_DIR=experiments/logs/e0_v2b bash experiments/e0_protocol_compare.sh quick
#     (new data -> new log dir, otherwise the crash-resume mistakes the old runs for done work)
#   TASKS=DTI CMP_MODELS=rgcn CMP_RUNS=3 bash experiments/e0_protocol_compare.sh
#   python experiments/protocol_compare_summary.py             # table (also run at the end)
#
# COST: runs = tasks x models x variants x CMP_RUNS (ladder, defaults: 2x2x6x3 = 72 trainings,
# each up to CMP_EPOCHS). Start with TASKS=DTI CMP_MODELS=rgcn if time is short.
# Crash-resume: a (task, model, variant) whose log already shows all runs completed is skipped.
# NOTE: CompGCN on TREATS with the full graph (s3_fullgraph, v2) may exceed 24 GB: failures are
# logged and the loop continues.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

MODE="${1:-ladder}"
TASKS="${TASKS:-DTI TREATS}"
CMP_MODELS="${CMP_MODELS:-rgcn compgcn}"
CMP_RUNS="${CMP_RUNS:-3}"
CMP_EPOCHS="${CMP_EPOCHS:-300}"
CMP_BASELINES="${CMP_BASELINES:-1}"
CMP_DISTMULT_LRS="${CMP_DISTMULT_LRS:-0.01 0.03 0.1}"
CMP_RESUME="${CMP_RESUME:-1}"
# CMP_DIR is where runs are logged AND where the crash-resume looks for completed runs:
# point it somewhere new when the DATA changed, or old logs will be mistaken for done work
CMP_DIR="${CMP_DIR:-${LOG_DIR}/e0}"
mkdir -p "$CMP_DIR"

# ---- variant flags: v1 flags + overrides (argparse keeps the LAST occurrence of a flag) ----
F_V1="$FLAGS_V1 --warm_eval"
F_FIX="$F_V1 --split_seed ${V2_SPLIT_SEED}"
F_S1="$F_FIX --select_metric mixed --patience ${PATIENCE_V2}"
# pinned to 5, not ${V2_TRAIN_NEG}: this ladder isolates PROTOCOL changes, so the rung must mean
# "5 explicit negatives" for every variant, not "whatever the config happened to tune"
F_S2="$F_S1 --oversample_rate 1 --train_negative_rate 5"
F_S3="$F_S2 --undersample_rate 1.0"
F_V2="$F_S3 --disjoint_supervision ${V2_DISJOINT}"

flags_of () {
  case "$1" in
    v1) echo "$F_V1" ;; v1_fixsplit) echo "$F_FIX" ;; s1_selectM) echo "$F_S1" ;;
    s2_negatives) echo "$F_S2" ;; s3_fullgraph) echo "$F_S3" ;; v2) echo "$F_V2" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

case "$MODE" in
  quick)  VARIANTS="v2" ;;                      # sanity check on a new target relation: v2 + baselines
  pair)   VARIANTS="v1 v2 v1_fixsplit" ;;
  ladder) VARIANTS="v1 v2 v1_fixsplit s1_selectM s2_negatives s3_fullgraph" ;;
  *) echo "usage: $0 [quick|pair|ladder]"; exit 1 ;;
esac

tsv_of () { [ "$1" = "$TASK_A" ] && echo "$TSV_A" || echo "$TSV_B"; }

done_already () {   # $1 = log glob prefix, $2 = number of runs
  [ "$CMP_RESUME" = "1" ] && grep -lq "Completed run $(( $2 - 1 ))/$2" "$1"_*.log 2>/dev/null
}

run_variant () {   # $1 task  $2 model  $3 variant  $4 config  $5.. flags
  local task="$1" model="$2" variant="$3" cfg="$4"; shift 4
  local prefix="${CMP_DIR}/e0_${task}_${model}_${variant}"
  if done_already "$prefix" "$CMP_RUNS"; then
    echo "[E0] skip ${task}/${model}/${variant} (already complete)"; return 0
  fi
  local log="${prefix}_$(date +%Y%m%d_%H%M%S).log"
  echo "[E0] ${task} | ${model} | ${variant} | config=${cfg} -> ${log}"
  # shellcheck disable=SC2068
  run_logged "$log" python train_and_eval.py --tsv "$(tsv_of "$task")" --task "$task" --model "$model" --config "$cfg" \
    --runs "$CMP_RUNS" --epochs "$CMP_EPOCHS" $@ \
    || echo "[E0] FAILED ${task}/${model}/${variant} (see ${log})"
}

echo "[E0] mode=$MODE tasks='$TASKS' models='$CMP_MODELS' runs=$CMP_RUNS epochs=$CMP_EPOCHS variants='$VARIANTS'"
echo "[E0] logs and crash-resume: $CMP_DIR (CMP_RESUME=$CMP_RESUME)"
for task in $TASKS; do
  # protocol comparison uses the v1-tuned configs on both sides (conservative for v2);
  # CMP_CONFIG overrides it, e.g. when checking a new target relation with the best config available
  cfg="${CMP_CONFIG:-$(PROTOCOL=v1 resolve_config "$task")}"

  if [ "$CMP_BASELINES" = "1" ]; then
    pop_prefix="${CMP_DIR}/e0_${task}_popularity_baseline"
    if done_already "$pop_prefix" 1; then
      echo "[E0] skip ${task}/popularity (already complete)"
    else
      log="${pop_prefix}_$(date +%Y%m%d_%H%M%S).log"
      echo "[E0] ${task} | popularity baseline -> ${log}"
      run_logged "$log" python experiments/baseline_popularity.py --tsv "$(tsv_of "$task")" --task "$task" \
        --split_seed "$V2_SPLIT_SEED" || echo "[E0] FAILED ${task}/popularity"
    fi
    # DistMult without message passing, v2 protocol (disjoint supervision is irrelevant without a graph)
    for lr in $CMP_DISTMULT_LRS; do
      run_variant "$task" distmult "v2_lr${lr}" "$cfg" $F_V2 --disjoint_supervision 0 --learning_rate "$lr"
    done
  fi

  for model in $CMP_MODELS; do
    for variant in $VARIANTS; do
      # shellcheck disable=SC2046
      run_variant "$task" "$model" "$variant" "$cfg" $(flags_of "$variant")
    done
  done
done

python experiments/protocol_compare_summary.py --logdir "$CMP_DIR" --out experiments \
  || echo "[E0] summary failed — run: python experiments/protocol_compare_summary.py"
echo "[E0] done. Table: experiments/protocol_compare_summary.md"
