#!/usr/bin/env bash
# =============================================================================
# E1 on the -v2b data, end to end: both tasks, all models, then the summary.
# =============================================================================
# One command for the run that produces the thesis table. It wraps
# e1_main_training.sh (one invocation per task x model, so nothing is duplicated here) and adds
# the three things a multi-hour unattended job needs:
#
#   1. PREFLIGHT. The tuned configs, the datasets and the post-HPO code fixes are checked BEFORE
#      anything trains. Finding out after six hours that PKT-TREATS-best-v2b was never written,
#      or that the server was never pulled, is the failure this prevents.
#   2. CRASH-RESUME. A (task, model) whose log already records its final run is skipped, so the
#      script can simply be relaunched after a crash, an OOM or a dropped connection.
#   3. FAULT TOLERANCE. One model failing (CompGCN on Task B is the OOM-prone case) does not stop
#      the others; the failure is recorded and reported at the end.
#
# COST: 12 seeds x 3 models x 2 tasks at 1500 epochs. Task A is roughly 3 h, Task B roughly 25 h in
# the worst case, less with early stopping (patience PATIENCE_V2). RUN IT UNDER tmux OR nohup.
#
# USAGE:
#   tmux new -s e1
#   bash experiments/e1_run_v2b.sh                  # everything
#   bash experiments/e1_run_v2b.sh A                # Task A only (the cheap one)
#   E1_MODELS="rgcn distmult" bash experiments/e1_run_v2b.sh B
#   E1_RESUME=0 bash experiments/e1_run_v2b.sh      # force a redo, ignoring existing logs
#   E1_DRY=1 bash experiments/e1_run_v2b.sh         # print the plan and the checks, train nothing
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

export PROTOCOL="${PROTOCOL:-v2}"
export EPOCHS="${EPOCHS:-1500}"
source experiments/config.sh

WHICH="${1:-AB}"
SUFFIX="${E1_SUFFIX:--v2b}"
E1_MODELS="${E1_MODELS:-$MODELS}"
E1_RESUME="${E1_RESUME:-1}"
E1_DRY="${E1_DRY:-0}"
CFG_A="PKT-${TASK_A}-best${SUFFIX}"
CFG_B="PKT-${TASK_B}-best${SUFFIX}"

case "$WHICH" in A|B|AB) ;; *) echo "usage: $0 [A|B|AB]"; exit 1 ;; esac

fail=0
note () { echo "  [!] $*"; fail=1; }

echo "== preflight =="

# -- the tuned configs must exist, with every model we are about to train --
for cfg in $([ "$WHICH" != B ] && echo "$CFG_A"; [ "$WHICH" != A ] && echo "$CFG_B"); do
  missing="$(python - "$cfg" "$E1_MODELS" <<'PY'
import json, sys
cfg, models = sys.argv[1], sys.argv[2].split()
try:
    d = json.load(open('src/models_params.json'))
except Exception as e:
    print(f"models_params.json unreadable ({e})"); raise SystemExit
if cfg not in d:
    print(f"config '{cfg}' absent"); raise SystemExit
absent = [m for m in models if m not in d[cfg]]
print(f"config '{cfg}' has no entry for: {', '.join(absent)}" if absent else "")
PY
)"
  if [ -n "$missing" ]; then
    note "$missing"
    note "    run: python experiments/get_best_hpo_config.py --task <TASK> --suffix=$SUFFIX --write"
  else
    echo "  [ok] $cfg  ($E1_MODELS)"
  fi
done

# -- the task graphs must exist --
[ "$WHICH" != B ] && { [ -f "$TSV_A" ] && echo "  [ok] $TSV_A" || note "missing $TSV_A (run analysis/06_build_subgraphs.py)"; }
[ "$WHICH" != A ] && { [ -f "$TSV_B" ] && echo "  [ok] $TSV_B" || note "missing $TSV_B (run analysis/06_build_subgraphs.py)"; }

# -- the post-HPO code fixes must be present: catches a server that was never pulled --
grep -q "resolve_train_negative_rate" train_and_eval.py \
  && echo "  [ok] tuned train_negative_rate is honoured" \
  || note "train_and_eval.py predates the train_negative_rate fix: E1 would train with a fixed 5 negatives instead of the value the HPO chose. git pull first."
case "$COMMON_FLAGS" in
  *--dedup_eval*) echo "  [ok] near-duplicate metrics enabled" ;;
  *) note "config.sh has no --dedup_eval in FLAGS_V2: the ChEBI near-duplicate columns will be missing. git pull first." ;;
esac
case "$COMMON_FLAGS" in
  *"--train_negative_rate auto"*) echo "  [ok] train_negative_rate = auto" ;;
  *) echo "  [--] train_negative_rate is pinned in FLAGS_V2 (V2_TRAIN_NEG=${V2_TRAIN_NEG}), not read from the config" ;;
esac

[ "$fail" = 1 ] && { echo; echo "preflight failed: nothing was trained."; exit 1; }

echo
echo "== plan =="
echo "  protocol=$PROTOCOL  epochs=$EPOCHS  seeds=$RUNS  models='$E1_MODELS'"
echo "  logs=$LOG_DIR   resume=$E1_RESUME"
[ "$WHICH" != B ] && echo "  Task A ($TASK_A) with $CFG_A   ~3 h"
[ "$WHICH" != A ] && echo "  Task B ($TASK_B) with $CFG_B   ~25 h worst case"
echo

# A (task, model) is done when an earlier log recorded its final run.
done_already () {   # $1 = task name, $2 = model
  [ "$E1_RESUME" = 1 ] || return 1
  grep -lq "Completed run $((RUNS - 1))/$RUNS" "${LOG_DIR}"/e1_"$1"_"$2"_*.log 2>/dev/null
}

failed=""
train_task () {   # $1 = A|B
  local t="$1" task cfgvar
  if [ "$t" = A ]; then task="$TASK_A"; else task="$TASK_B"; fi
  for m in $E1_MODELS; do
    if done_already "$task" "$m"; then
      echo "[E1] skip $task/$m (already complete; E1_RESUME=0 to redo)"; continue
    fi
    echo "[E1] === $task / $m ==="
    if [ "$E1_DRY" = 1 ]; then echo "[E1] (dry run: would train $task/$m)"; continue; fi
    if [ "$t" = A ]; then
      TASKS=A MODELS="$m" CFG_A="$CFG_A" bash experiments/e1_main_training.sh
    else
      TASKS=B MODELS="$m" CFG_B="$CFG_B" bash experiments/e1_main_training.sh
    fi
    [ $? -ne 0 ] && { echo "[E1] FAILED $task/$m (continuing)"; failed="$failed $task/$m"; }
  done
}

[ "$WHICH" != B ] && train_task A
[ "$WHICH" != A ] && train_task B

echo
if [ "$E1_DRY" = 1 ]; then echo "[E1] dry run finished, nothing trained."; exit 0; fi
python experiments/e1_summary.py || echo "[E1] summary failed: rerun python experiments/e1_summary.py"
[ -n "$failed" ] && echo "[E1] models that failed:$failed"
echo "[E1] done. Logs in $LOG_DIR ; table in ${LOG_DIR}/e1_summary.md"
