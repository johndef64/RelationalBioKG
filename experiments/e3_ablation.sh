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
#   v1 -> legacy behaviour (compgcn, 3 seeds, 200 epochs)
#   v2 -> FLAGS_V2 + one override per tag, ABL_RUNS=5 seeds (fixed split: variance = init only),
#         model ABL_MODEL (default rgcn), task ABL_TASK (A or B, both families),
#         at most ABL_EPOCHS=1500 epochs (the E1 budget; early stopping ends most runs well before)
#
# OUTPUT — everything a run produces goes into ONE versioned folder, nothing into models/ or logs/:
#
#   experiments/ablation/<TASK>_v<N>/            e.g. experiments/ablation/DTI_v2/
#     MANIFEST.md   when, where (host, GPU), code commit, settings; appended on every resume
#     settings.env  the settings the version was started with: a resume with different ones is refused
#     logs/         e3_<tag>_<timestamp>.log, one per variant (and per attempt, if one crashed)
#     models/<tag>/ params, metrics and the checkpoints rgcn_run<i>.pt of that variant
#     summary/      ablation_summary.md / .csv, regenerated at the end of every launch
#     COMPLETE      written when every variant of the launch has all its seeds
#
#   Version choice: ABL_VERSION=v<N> forces one. Otherwise the latest version of the task is RESUMED
#   if it has no COMPLETE marker, and a new one (latest+1) is started if it has. ABL_NEW=1 always
#   starts a new version. A version is never shared by two sets of settings, so two runs of one
#   version cannot differ in anything but the variant: that is what makes their deltas readable.
#
# DETERMINISM: ABL_DETERMINISTIC=1 (default) passes --deterministic, under which two executions of
# the same variant and seed are bit-identical (src/deterministic_ops.py). DTI_v1 predates it.
#
# Usage:  PROTOCOL=v2 bash experiments/e3_ablation.sh                      # v2, Task A, both families
#         PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh component # v2, Task B, component only
#         PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh context   # v2, Task B, context only
#         ABL_DRY=1 PROTOCOL=v2 bash experiments/e3_ablation.sh            # print the plan, train nothing
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

WHICH="${1:-all}"
ABL_RESUME="${ABL_RESUME:-1}"   # 1 = crash-resume: skip tags already fully completed. Set 0 to force redo.
ABL_DETERMINISTIC="${ABL_DETERMINISTIC:-1}"
ABL_DRY="${ABL_DRY:-0}"
ABL_ROOT="${ABL_ROOT:-experiments/ablation}"

if [ "$PROTOCOL" = "v2" ]; then
  # 1500 = the E1 budget. It used to default to $EPOCHS (800 under v2), which is below the best
  # epoch of some E1 Task A seeds (up to 925): the ablation would have been cut short where E1 was not,
  # and every delta against it would have mixed the ablated factor with a smaller training budget.
  ABL_RUNS="${ABL_RUNS:-5}"; ABL_EPOCHS="${ABL_EPOCHS:-1500}"
  MODEL="${ABL_MODEL:-rgcn}"
  ABL_TASK="${ABL_TASK:-A}"
  if [ "$ABL_TASK" = "B" ]; then A_TSV="$TSV_B"; A_TASK="$TASK_B"; else A_TSV="$TSV_A"; A_TASK="$TASK_A"; fi
  VPREFIX="${A_TASK}"
else
  ABL_RUNS="${RUNS:-3}"; ABL_EPOCHS="${EPOCHS:-200}"
  MODEL="${ABL_MODEL:-compgcn}"
  A_TSV="$TSV_A"; A_TASK="$TASK_A"
  VPREFIX="${A_TASK}_protocolv1"
fi
CFG="${ABL_CONFIG:-$(resolve_config "$A_TASK")}"
DET_FLAG=""; [ "$ABL_DETERMINISTIC" = "1" ] && DET_FLAG="--deterministic"

# ---- which version folder ----
latest_version () {   # prints the highest N of <prefix>_v<N>, or 0
  local n=0 d v
  for d in "$ABL_ROOT/${VPREFIX}"_v*; do
    [ -d "$d" ] || continue
    v="${d##*_v}"; [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -gt "$n" ] && n="$v"
  done
  echo "$n"
}
if [ -n "${ABL_VERSION:-}" ]; then
  VERSION="$ABL_VERSION"; WHY="forced by ABL_VERSION"
else
  LAST="$(latest_version)"
  if [ "${ABL_NEW:-0}" != "1" ] && [ "$LAST" -gt 0 ] && [ ! -f "$ABL_ROOT/${VPREFIX}_v${LAST}/COMPLETE" ]; then
    VERSION="v${LAST}"; WHY="resuming: the latest version has no COMPLETE marker"
  else
    VERSION="v$((LAST + 1))"; WHY="new version"
  fi
fi
ABL_DIR="$ABL_ROOT/${VPREFIX}_${VERSION}"
ABL_LOG_DIR="$ABL_DIR/logs"
SETTINGS="PROTOCOL=$PROTOCOL MODEL=$MODEL CONFIG=$CFG RUNS=$ABL_RUNS EPOCHS=$ABL_EPOCHS DETERMINISTIC=$ABL_DETERMINISTIC TSV=$A_TSV"

echo "[E3] version $ABL_DIR ($WHY)"
echo "[E3] $SETTINGS"
if [ "$ABL_DRY" = "1" ]; then
  echo "[E3] dry run: nothing is written or trained"
else
  mkdir -p "$ABL_LOG_DIR" "$ABL_DIR/models" "$ABL_DIR/summary"
  if [ -f "$ABL_DIR/settings.env" ]; then
    if [ "$(cat "$ABL_DIR/settings.env")" != "$SETTINGS" ]; then
      echo "[E3] ERROR: $ABL_DIR was started with different settings:"
      echo "       then: $(cat "$ABL_DIR/settings.env")"
      echo "       now : $SETTINGS"
      echo "     Resume with the same settings, or start a new version with ABL_NEW=1."
      exit 1
    fi
  else
    echo "$SETTINGS" > "$ABL_DIR/settings.env"
  fi
  # the pre-registration is frozen into the version when the version is created, before any result.
  # It lives in experiments/prereg/ (tracked by git, so the commit date proves when it was written);
  # experiments/ablation/ itself is not in git.
  PREREG="${ABL_PREREG_DIR:-experiments/prereg}/PREREGISTRATION_${A_TASK}.json"
  if [ -f "$PREREG" ] && [ ! -f "$ABL_DIR/PREREGISTRATION.json" ]; then
    cp "$PREREG" "$ABL_DIR/PREREGISTRATION.json"; echo "[E3] pre-registration frozen into $ABL_DIR/"
  fi
  # under Slurm the driver's own output (skips, warnings, final status) also belongs to the version
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[E3] Slurm job $SLURM_JOB_ID: driver output -> $ABL_DIR/driver.log"
    exec >> "$ABL_DIR/driver.log" 2>&1
    echo "===== $(date '+%F %T') Slurm job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
  fi
  GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  {
    [ -s "$ABL_DIR/MANIFEST.md" ] || printf '# E3 ablation — %s, %s\n\n| launch | host | GPU | commit | what |\n|---|---|---|---|---|\n' "$A_TASK" "$VERSION"
    printf '| %s | %s | %s | %s%s | %s |\n' "$(date '+%F %T')" "$(hostname)" "${GPU:-n/a}" \
      "$(git rev-parse --short HEAD 2>/dev/null || echo n/a)" \
      "$(git diff --quiet 2>/dev/null || echo ' (+ local changes)')" "$WHICH ($WHY)"
  } >> "$ABL_DIR/MANIFEST.md"
fi

# A tag counts as "done" if an earlier log of THIS version recorded its final run (Completed run N-1/N).
# Only FULLY completed tags are skipped; a tag that crashed mid-way is re-run from scratch.
tag_done () {   # $1 = tag
  local last=$((ABL_RUNS - 1))
  grep -lq "Completed run ${last}/${ABL_RUNS}" "${ABL_LOG_DIR}"/e3_"$1"_*.log 2>/dev/null
}

ALL_DONE=1
train () {  # $1=tag  $2=tsv  ... extra flags
  local tag="$1" tsv="$2"; shift 2
  if [ "$ABL_RESUME" = "1" ] && tag_done "$tag"; then
    echo "[E3] skip '$tag' — already completed (found in ${ABL_LOG_DIR}/e3_${tag}_*.log)"; return 0
  fi
  local log="${ABL_LOG_DIR}/e3_${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo "[E3] $tag -> $log"
  local out=(--models_dir "$ABL_DIR/models" --run_name "$tag")
  if [ "$ABL_DRY" = "1" ]; then
    local proto_flags="$FLAGS_V2"; [ "$PROTOCOL" = "v2" ] || proto_flags="--early_stopping --patience $PATIENCE --eval_filtered"
    echo "      python train_and_eval.py --tsv $tsv --task $A_TASK --model $MODEL --config $CFG --runs $ABL_RUNS --epochs $ABL_EPOCHS $proto_flags $DET_FLAG ${out[*]} $*"
    return 0
  fi
  if [ "$PROTOCOL" = "v2" ]; then
    # shellcheck disable=SC2086
    run_logged "$log" python train_and_eval.py --tsv "$tsv" --task "$A_TASK" --model "$MODEL" --config "$CFG" \
      --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS" $FLAGS_V2 $DET_FLAG "${out[@]}" "$@" || true
  else
    # shellcheck disable=SC2086
    run_logged "$log" python train_and_eval.py --tsv "$tsv" --task "$A_TASK" --model "$MODEL" --config "$CFG" \
      --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS" --early_stopping --patience "$PATIENCE" --eval_filtered \
      $DET_FLAG "${out[@]}" "$@" || true
  fi
  tag_done "$tag" || { ALL_DONE=0; echo "[E3] WARNING: '$tag' did not complete all $ABL_RUNS seeds — see $log"; }
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
  # NB ctx_full is trained on the ablation "full" file, which holds the same rows IN THE SAME ORDER as
  # the task subgraph (same content hash, checked 2026-09-24): it is the same experiment as comp_full.
  # An older comment here claimed a different row order, hence a different split: that was wrong.
  # It is kept as a separate run so that each family has its own reference in the summary; under
  # --deterministic the two are bit-identical, which doubles as a check that nothing drifted.
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

[ "$ABL_DRY" = "1" ] && exit 0
python experiments/ablation_summary.py --logdir "$ABL_LOG_DIR" --out "$ABL_DIR/summary" || true
if [ "$ALL_DONE" = "1" ] && [ "$WHICH" = "all" ]; then
  date '+%F %T' > "$ABL_DIR/COMPLETE"
  echo "[E3] $ABL_DIR complete. Summary in $ABL_DIR/summary/"
else
  echo "[E3] $ABL_DIR NOT complete (a variant failed, or only '$WHICH' was run)."
  echo "     Relaunch the same command to resume this version; the summary so far is in $ABL_DIR/summary/"
fi
