#!/usr/bin/env bash
# Shared configuration for all PKT experiments. Sourced by the e*.sh scripts.
# Edit here once; every experiment picks it up.

# ---- conda env ----
export CONDA_ENV="${CONDA_ENV:-gnn}"

# ---- reproducibility ----
# Python randomises str hashes per process: the node-type sets built in src/utils.py then iterate
# in a different order at every launch, which changes the order in which the per-type embedding
# tables are created and therefore their random init -> identical commands gave different results
# (verified on CompGCN, even on CPU single-thread). Fixing the hash seed makes runs reproducible.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# ---- datasets (built by analysis/06_build_subgraphs.py) ----
export TSV_A="dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip"      # Task A: predict DTI
export TSV_B="dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip"   # Task B: predict TREATS

# ---- task target relation names (interaction column) ----
export TASK_A="DTI"
export TASK_B="TREATS"

# ---- compound-centric target node type (drug_eval --target_type) ----
export TGT_A="Protein"
export TGT_B="Disease"

# ---- model / hyperparameter config (key in src/models_params.json) ----
# HP_CONFIG is the FALLBACK config used when no tuned PKT-<TASK>-best exists yet.
# BIOKG-128 = DRKG-scale biomedical config (right scale for PKT ~66-94k nodes).
export HP_CONFIG="${HP_CONFIG:-BIOKG-128}"
# MODELS default depends on PROTOCOL (set below): v1 = compgcn rgcn ; v2 = rgcn compgcn distmult

# Auto-select the tuned config PKT-<TASK>-best from src/models_params.json when present,
# otherwise fall back to $HP_CONFIG. Set USE_BEST=0 to always use $HP_CONFIG (e.g. baselines).
export USE_BEST="${USE_BEST:-1}"
resolve_config () {   # $1 = task interaction (DTI / TREATS); echoes the config name to use
  local want="PKT-$1-best" s found=0
  # Protocol v2 prefers configs tuned under v2, newest suffix first: -v2b is the HPO on the final
  # data (with the DrugBank DTI layer, analysis/10_build_dti_drugbank.py) used for all results.
  if [ "${PROTOCOL:-v1}" = "v2" ]; then
    for s in ${V2_CONFIG_SUFFIXES:--v2b -v2}; do
      if python -c "import json,sys;sys.exit(0 if 'PKT-$1-best$s' in json.load(open('src/models_params.json')) else 1)" 2>/dev/null; then
        want="PKT-$1-best$s"; found=1; break
      fi
    done
    [ "$found" = 1 ] || echo "[config] WARNING: PROTOCOL=v2 but no PKT-$1-best{${V2_CONFIG_SUFFIXES:--v2b -v2}} config: falling back to $want (not tuned under v2)" >&2
  fi
  if [ "$USE_BEST" = "1" ] && \
     python -c "import json,sys;sys.exit(0 if '$want' in json.load(open('src/models_params.json')) else 1)" 2>/dev/null; then
    echo "$want"
  else
    echo "$HP_CONFIG"
  fi
}

# ---- training budget (PathogenKG Table-4 protocol) ----
export RUNS="${RUNS:-12}"
# v2: early stopping on val M (patience 50) is the real stop; in E0 the v1-tuned R-GCN still had
# its best epoch at 292/300, so the v2 ceiling is higher to avoid truncating slow-converging GNNs.
if [ "${PROTOCOL:-v1}" = "v2" ]; then export EPOCHS="${EPOCHS:-800}"; else export EPOCHS="${EPOCHS:-400}"; fi
export PATIENCE="${PATIENCE:-20}"

# ---- training protocols ----
# v1 = PathogenKG protocol: x5 oversampling, 50% random background undersampling, split
#      changing at every run, checkpoint/early stopping on validation LOSS, training target edges
#      inside the message-passing graph.
# v2 = consolidated protocol: fixed split, checkpoint/early stopping on validation M (as the HPO),
#      no oversampling + explicit training negatives, full context graph, disjoint supervision
#      edges, cold-start-excluded test metrics reported alongside.
# Every v2 change is a train_and_eval.py flag whose default reproduces v1.
export PATIENCE_V2="${PATIENCE_V2:-50}"                 # epochs without val-M improvement
export V2_SPLIT_SEED="${V2_SPLIT_SEED:-42}"             # = split used by the HPO
# 'auto' = the value the HPO tuned for this model, read back from the config (see
# resolve_train_negative_rate in train_and_eval.py); configs with no tuned value fall back to 5.
export V2_TRAIN_NEG="${V2_TRAIN_NEG:-auto}"
export V2_DISJOINT="${V2_DISJOINT:-0.3}"

# CUDA allocator: keep one arena that can grow instead of many fixed blocks. Full-batch training on
# the large context graph allocates a few large tensors per epoch, which otherwise fragments the pool
# and causes out-of-memory errors with free memory still reserved.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export FLAGS_V1="--early_stopping --patience ${PATIENCE} --negative_sampling filtered --eval_filtered \
--oversample_rate 5 --undersample_rate 0.5 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0"
export FLAGS_V2="--early_stopping --patience ${PATIENCE_V2} --negative_sampling filtered --eval_filtered \
--oversample_rate 1 --undersample_rate 1.0 --alpha 0.25 --gamma 3.0 --alpha_adv 2.0 \
--split_seed ${V2_SPLIT_SEED} --select_metric mixed --train_negative_rate ${V2_TRAIN_NEG} \
--disjoint_supervision ${V2_DISJOINT} --warm_eval --dedup_eval"

# PROTOCOL selects the flags used by E1/E3/E4: v1 reproduces the PathogenKG protocol, v2 is the
# consolidated protocol used for all reported results (with the PKT-<TASK>-best-v2b configs).
export PROTOCOL="${PROTOCOL:-v1}"
if [ "$PROTOCOL" = "v2" ]; then
  export COMMON_FLAGS="$FLAGS_V2"
  export MODELS="${MODELS:-rgcn compgcn distmult}"   # two GNNs + embedding-only DistMult baseline
else
  export COMMON_FLAGS="$FLAGS_V1"
  export MODELS="${MODELS:-compgcn rgcn}"
fi

# ---- logging ----
# v2 logs go to their own folder so they never mix with v1 logs (E1/E3 summaries glob by name)
if [ "${PROTOCOL:-v1}" = "v2" ]; then
  export LOG_DIR="${LOG_DIR:-experiments/logs/v2}"
else
  export LOG_DIR="${LOG_DIR:-experiments/logs}"
fi
mkdir -p "$LOG_DIR"

# ---- running a long job without letting the terminal freeze it ----
# "cmd 2>&1 | tee LOG" writes to the terminal as well as to the log, which ties the job's progress
# to someone draining that terminal: if the terminal stops reading (e.g. a disconnected remote
# session), tee blocks inside write() and the training process blocks with it, holding the GPU.
#
# run_logged sends the job's own output straight to the file, where nothing can block it, and lets
# a separate tail do the talking to the terminal: if the terminal stalls, only the tail stalls.
# Long jobs should still run under nohup or a batch scheduler, which also survive a closed terminal.
run_logged () {          # run_logged <logfile> <command...>
  local log="$1"; shift
  : > "$log"
  "$@" > "$log" 2>&1 &
  local pid=$!
  tail -n +1 -f "$log" --pid="$pid" 2>/dev/null &
  local tpid=$!
  local rc=0
  wait "$pid" || rc=$?              # plain "wait" under set -e would abort before we read $?
  _reap_tail "$tpid"
  return "$rc"
}

# Give the tail up to 3 s to print what is left and exit by itself (--pid makes it check once a
# second), then kill it. Killing it immediately loses the output of short commands; waiting for it
# unconditionally would block the same way, since a tail writing to a stalled terminal blocks in
# write() exactly as tee does.
_reap_tail () {
  local tpid="$1" waited=0
  while kill -0 "$tpid" 2>/dev/null && [ "$waited" -lt 3 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  kill "$tpid" 2>/dev/null || true
  wait "$tpid" 2>/dev/null || true
}

run_logged_append () {   # same, but appending to an existing log
  local log="$1"; shift
  "$@" >> "$log" 2>&1 &
  local pid=$!
  tail -n 0 -f "$log" --pid="$pid" 2>/dev/null &
  local tpid=$!
  local rc=0
  wait "$pid" || rc=$?
  _reap_tail "$tpid"
  return "$rc"
}

# Activate conda env if available (harmless if already active). Set PKT_SKIP_CONDA=1 when the
# environment is already activated, e.g. by a Slurm job.
if [ "${PKT_SKIP_CONDA:-0}" != "1" ] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

echo "[config] env=$CONDA_ENV protocol=$PROTOCOL config=$HP_CONFIG runs=$RUNS epochs=$EPOCHS models='$MODELS'"
