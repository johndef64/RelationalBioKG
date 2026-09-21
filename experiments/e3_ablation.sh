#!/usr/bin/env bash
# E3 — Ablation studies (adapted from PathogenKG). Two families:
#
#   E3a  COMPONENT ablation — turn off, one at a time, a piece of the training machinery.
#        v1 (legacy): focal off · adversarial off · no oversampling · no undersampling · random negatives
#        v2         : focal off · adversarial off · 1 training negative (k=1) · no disjoint supervision
#                     · 50% random background undersampling (control documenting the v2 choice)
#
#   E3b  RELATIONAL-CONTEXT ablation — train on subgraphs that drop one context layer at a time
#        (built by analysis/07_build_ablation_subgraphs.py, listed in ablation_index_{A,B}.md):
#          Task A: full / core_ppi / no_ppi / no_go / no_pathway / no_drugctx / no_biochem
#          Task B: full / no_ppi / no_go / no_pathway / no_biochem / no_pharma / no_disease_ctx
#
# PROTOCOL (experiments/config.sh):
#   v1 -> legacy behaviour, logs in experiments/logs/e3_<tag>_<ts>.log
#   v2 -> FLAGS_V2 + one override per tag, ABL_RUNS=5 seeds (fixed split: variance = init only),
#         model ABL_MODEL (default rgcn), task ABL_TASK (A or B, both families),
#         at most ABL_EPOCHS=1500 epochs (the E1 budget; early stopping ends most runs well before),
#         logs in experiments/logs/v2/e3_<TASK>/e3_<tag>_<ts>.log  (never mixed with v1)
#
# Usage:  bash experiments/e3_ablation.sh                                  # v1, both families
#         PROTOCOL=v2 bash experiments/e3_ablation.sh                      # v2, Task A, both families
#         PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh component # v2, Task B, component only
#         PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh context   # v2, Task B, context only
# Summary: python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_DTI --out experiments/logs/v2/e3_DTI
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

WHICH="${1:-all}"
ABL_RESUME="${ABL_RESUME:-1}"   # 1 = crash-resume: skip tags already fully completed. Set 0 to force redo.

if [ "$PROTOCOL" = "v2" ]; then
  # 1500 = the E1 budget. It used to default to $EPOCHS (800 under v2), which is below the best
  # epoch of some E1 Task A seeds (up to 925): the ablation would have been cut short where E1 was not,
  # and every delta against it would have mixed the ablated factor with a smaller training budget.
  ABL_RUNS="${ABL_RUNS:-5}"; ABL_EPOCHS="${ABL_EPOCHS:-1500}"
  MODEL="${ABL_MODEL:-rgcn}"
  ABL_TASK="${ABL_TASK:-A}"
  if [ "$ABL_TASK" = "B" ]; then A_TSV="$TSV_B"; A_TASK="$TASK_B"; else A_TSV="$TSV_A"; A_TASK="$TASK_A"; fi
  ABL_LOG_DIR="${LOG_DIR}/e3_${A_TASK}"
else
  ABL_RUNS="${RUNS:-3}"; ABL_EPOCHS="${EPOCHS:-200}"
  MODEL="${ABL_MODEL:-compgcn}"
  A_TSV="$TSV_A"; A_TASK="$TASK_A"
  ABL_LOG_DIR="${LOG_DIR}"
fi
mkdir -p "$ABL_LOG_DIR"
echo "[E3] protocol=$PROTOCOL task=$A_TASK model=$MODEL runs=$ABL_RUNS epochs=$ABL_EPOCHS logs=$ABL_LOG_DIR"

# A tag counts as "done" if an earlier log recorded its final run (Completed run N-1/N).
# Only FULLY completed tags are skipped; a tag that crashed mid-way is re-run from scratch.
tag_done () {   # $1 = tag
  local last=$((ABL_RUNS - 1))
  grep -lq "Completed run ${last}/${ABL_RUNS}" "${ABL_LOG_DIR}"/e3_"$1"_*.log 2>/dev/null
}

train () {  # $1=tag  $2=tsv  ... extra flags
  local tag="$1" tsv="$2"; shift 2
  if [ "$ABL_RESUME" = "1" ] && tag_done "$tag"; then
    echo "[E3] skip '$tag' — already completed (found in ${ABL_LOG_DIR}/e3_${tag}_*.log)"; return 0
  fi
  local log="${ABL_LOG_DIR}/e3_${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo "[E3] $tag -> $log"
  # ABL_CONFIG forces a config name (e.g. a differently-suffixed HPO such as PKT-DTI-best-v2b);
  # otherwise v2 resolves PKT-<TASK>-best-v2 and v1 PKT-<TASK>-best
  local cfg; cfg="${ABL_CONFIG:-$(resolve_config "$A_TASK")}"
  if [ "$PROTOCOL" = "v2" ]; then
    # shellcheck disable=SC2086
    run_logged "$log" python train_and_eval.py --tsv "$tsv" --task "$A_TASK" --model "$MODEL" --config "$cfg" \
      --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS" $FLAGS_V2 "$@"
  else
    run_logged "$log" python train_and_eval.py --tsv "$tsv" --task "$A_TASK" --model "$MODEL" --config "$cfg" \
      --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS" --early_stopping --patience "$PATIENCE" --eval_filtered "$@"
  fi
}

component_ablation () {
  echo "== E3a component ablation (Task $A_TASK) =="
  if [ "$PROTOCOL" = "v2" ]; then
    train "comp_full"          "$A_TSV"                                   # all v2 machinery on (reference)
    train "comp_no_focal"      "$A_TSV" --alpha 1.0 --gamma 0.0           # focal loss off
    train "comp_no_adv"        "$A_TSV" --alpha_adv 0.0                   # adversarial negative weighting off
    train "comp_neg1"          "$A_TSV" --train_negative_rate 1           # 1 negative per positive
    train "comp_no_disjoint"   "$A_TSV" --disjoint_supervision 0          # target edges back in the graph
    train "comp_undersample05" "$A_TSV" --undersample_rate 0.5            # legacy 50% random background
    return
  fi
  # ---- v1 (legacy) ----
  # full reference (all machinery on)
  train "comp_full"        "$A_TSV" --negative_sampling filtered --oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
  # focal loss OFF (plain BCE-like: alpha=1, gamma=0)
  train "comp_no_focal"    "$A_TSV" --negative_sampling filtered --oversample_rate 5 --undersample_rate 0.5 --alpha 1.0 --gamma 0.0 --alpha_adv 2.0
  # adversarial negative weighting OFF
  train "comp_no_adv"      "$A_TSV" --negative_sampling filtered --oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 0.0
  # no oversampling of positives
  train "comp_no_oversmp"  "$A_TSV" --negative_sampling filtered --oversample_rate 1 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
  # no undersampling of background
  train "comp_no_undersmp" "$A_TSV" --negative_sampling filtered --oversample_rate 5 --undersample_rate 1.0 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
  # random negatives instead of filtered
  train "comp_rand_neg"    "$A_TSV" --negative_sampling standard --oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
}

context_ablation () {
  local abl="dataset/PKT_subgraphs/ablation" pfx variants
  if [ "$A_TASK" = "$TASK_A" ]; then
    pfx="ablA"; variants="full core_ppi no_ppi no_go no_pathway no_drugctx no_biochem"
  else
    [ "$PROTOCOL" = "v2" ] || { echo "[E3b] Task B context ablation is v2 only — skipped"; return 0; }
    pfx="ablB"; variants="full no_ppi no_go no_pathway no_biochem no_pharma no_disease_ctx"
  fi
  echo "== E3b relational-context ablation (Task $A_TASK) =="
  if [ ! -f "$abl/pkt_${pfx}_full.tsv.zip" ]; then
    echo "[E3b] building ablation subgraphs..."; python analysis/07_build_ablation_subgraphs.py
  fi
  # NB ctx_full is trained on the ablation "full" file even though it holds the same triples as the
  # task subgraph: the rows are in a different order and the stratified split depends on row order,
  # so only the files of one ablation family share the same split with each other. The context
  # deltas are therefore read against ctx_full, never against the E1 numbers.
  for v in $variants; do
    if [ "$PROTOCOL" = "v2" ]; then
      train "ctx_${v}" "$abl/pkt_${pfx}_${v}.tsv.zip"
    else
      train "ctx_${v}" "$abl/pkt_${pfx}_${v}.tsv.zip" \
        --oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0 --negative_sampling filtered
    fi
  done
}

case "$WHICH" in
  component) component_ablation ;;
  context)   context_ablation ;;
  all)       component_ablation; context_ablation ;;
  *) echo "usage: $0 [component|context|all]"; exit 1 ;;
esac
echo "[E3] done. Logs in ${ABL_LOG_DIR}/ ; summary: python experiments/ablation_summary.py --logdir ${ABL_LOG_DIR} --out ${ABL_LOG_DIR}"
