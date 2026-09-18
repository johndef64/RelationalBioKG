#!/usr/bin/env bash
# =============================================================================
# E1 — progress report. READ-ONLY: it only reads log files, so it cannot disturb a running job.
#
# Reads the markers train_and_eval.py writes into each E1 log:
#   "[val] run i epoch N ..."   every evaluate_every epochs  -> where the current run is
#   "[i] Completed run i/N"     at the end of each seed       -> how many seeds are done
#   "Run i | best_epoch: ..."   per seed                      -> whether early stopping fired
#
# Usage:
#   bash experiments/e1_progress.sh              # one report and exit
#   bash experiments/e1_progress.sh -w           # refresh every 60 s (Ctrl-C to leave; the job
#                                                #  keeps running, this only stops the watching)
#   bash experiments/e1_progress.sh -w 15        # refresh every 15 s
#   LOG_DIR=experiments/logs/v2 bash experiments/e1_progress.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="${LOG_DIR:-experiments/logs/v2}"
RUNS="${RUNS:-12}"
EPOCHS="${EPOCHS:-1500}"
STALE_MIN="${STALE_MIN:-10}"     # a log untouched for longer than this is reported as idle

report () {
  local logs
  logs=$(ls -t "$LOG_DIR"/e1_*.log 2>/dev/null) || true
  if [ -z "$logs" ]; then
    echo "no E1 log in $LOG_DIR (set LOG_DIR= if the run used another directory)"
    return
  fi

  printf "%-26s %-9s %-8s %-22s %-9s %s\n" "LOG" "STATE" "SEEDS" "CURRENT SEED" "LAST" "BEST EPOCHS"
  printf "%s\n" "----------------------------------------------------------------------------------------------------"

  local now; now=$(date +%s)
  for log in $logs; do
    local base done_runs last_val cur_run cur_ep mtime age state best
    base=$(basename "$log" .log)
    base=${base#e1_}

    # grep -c already prints 0 and exits 1 when there is no match: adding a fallback duplicates it
    done_runs=$(grep -c "Completed run" "$log" 2>/dev/null); done_runs=${done_runs:-0}
    last_val=$(grep "^\[val\] run" "$log" 2>/dev/null | tail -1)
    cur_run=$(sed -n 's/^\[val\] run \([0-9]*\) epoch.*/\1/p' <<<"$last_val")
    cur_ep=$(sed -n 's/^\[val\] run [0-9]* epoch \([0-9]*\).*/\1/p' <<<"$last_val")

    mtime=$(stat -c %Y "$log" 2>/dev/null || stat -f %m "$log" 2>/dev/null || echo "$now")
    age=$(( (now - mtime) / 60 ))

    if grep -qE "Traceback|CUDA out of memory|Killed" "$log" 2>/dev/null; then state="FAILED"
    elif [ "$done_runs" -ge "$RUNS" ]; then   state="DONE"
    elif [ "$age" -ge "$STALE_MIN" ];   then  state="idle ${age}m"
    else                                      state="running"; fi

    # best_epoch of the finished seeds: at the ceiling means the epoch budget is binding
    best=$(sed -n 's/.*best_epoch: \([0-9]*\).*/\1/p' "$log" 2>/dev/null | tr '\n' ' ')
    [ -z "$best" ] && best="-"

    printf "%-26s %-9s %-8s %-22s %-9s %s\n" \
      "${base:0:26}" "$state" "${done_runs}/${RUNS}" \
      "$( [ -n "$cur_run" ] && echo "seed $cur_run, ep $cur_ep/$EPOCHS" || echo "-" )" \
      "${age}m ago" "${best:0:34}"
  done

  echo
  # last test result seen anywhere, so the numbers start appearing before the whole thing ends
  local last_test
  last_test=$(grep -h "^Run .* | Test Auroc" $logs 2>/dev/null | tail -3)
  if [ -n "$last_test" ]; then
    echo "ultimi risultati di test comparsi nei log:"
    sed 's/^/    /' <<<"$last_test"
  fi

  # anything that died
  local fails
  fails=$(grep -lE "Traceback|CUDA out of memory|Killed" $logs 2>/dev/null || true)
  [ -n "$fails" ] && { echo; echo "!! log con errori: $(echo $fails | tr '\n' ' ')"; }
}

if [ "${1:-}" = "-w" ]; then
  every="${2:-60}"
  while true; do
    clear
    echo "E1 progress — $(date '+%H:%M:%S')   (Ctrl-C esce dal monitor, NON ferma il training)"
    echo
    report
    sleep "$every"
  done
else
  report
fi
