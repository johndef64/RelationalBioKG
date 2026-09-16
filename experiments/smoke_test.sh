#!/usr/bin/env bash
# =============================================================================
# SMOKE TEST — does every experiment still run after a dataset change?
# =============================================================================
# Runs one tiny instance of every step of the pipeline (2-4 epochs, 1 run, small graphs) and
# reports PASS/FAIL per step. It checks that things WORK, not that they work well: the metrics
# it prints are meaningless.
#
# Run it after rebuilding the task graphs (analysis/06_build_subgraphs.py) and before launching
# anything long, locally or on the server.
#
# USAGE:
#   bash experiments/smoke_test.sh              # everything except the W&B sweep
#   SMOKE_GPU=1 bash experiments/smoke_test.sh  # use the GPU (default: CPU, safer on a laptop)
#   SMOKE_HPO=1 bash experiments/smoke_test.sh  # also try one HPO trial (needs wandb login)
#
# Exit code 0 = all steps passed.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
source experiments/config.sh

SMOKE_DIR="${LOG_DIR}/smoke"
mkdir -p "$SMOKE_DIR"
[ "${SMOKE_GPU:-0}" = "1" ] || export CUDA_VISIBLE_DEVICES=-1
EP="${SMOKE_EPOCHS:-4}"
PASS=0; FAIL=0; SKIP=0
RESULTS=()

step () {   # $1 = name, rest = command
  local name="$1"; shift
  local log="${SMOKE_DIR}/$(echo "$name" | tr ' /' '__').log"
  printf '%-42s' "[smoke] $name ..."
  if "$@" > "$log" 2>&1; then
    echo " PASS"; PASS=$((PASS+1)); RESULTS+=("PASS  $name")
  else
    echo " FAIL  -> $log"; FAIL=$((FAIL+1)); RESULTS+=("FAIL  $name  ($log)")
    tail -5 "$log" | sed 's/^/        /'
  fi
}

TRAIN_FLAGS="--negative_sampling filtered --eval_filtered --oversample_rate 1 --undersample_rate 1.0 \
--split_seed ${V2_SPLIT_SEED} --select_metric mixed --train_negative_rate 1 \
--disjoint_supervision ${V2_DISJOINT} --warm_eval --evaluate_every 2"

echo "[smoke] env=$CONDA_ENV  device=$([ "${SMOKE_GPU:-0}" = 1 ] && echo GPU || echo CPU)  epochs=$EP  logs=$SMOKE_DIR"
echo

# ---- 1. data: the task graphs load and the target relation is found ---------
step "data: task graphs load" python - <<'PY'
import sys
sys.path.insert(0, ".")
from src.utils import load_data, set_target_label
expected = {"DTI": "dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip",
            "TREATS": "dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip"}
for task, tsv in expected.items():
    df = load_data(tsv, {}, quiet=True)
    df = df[0] if isinstance(df, tuple) else df
    lab = set_target_label(df.copy(), [task])
    lab = lab[0] if isinstance(lab, tuple) else lab
    n = int(lab["label"].sum())
    rels = sorted(df["interaction"].unique())
    print(f"{task}: {len(df):,} edges, {n:,} target edges, relations={rels}")
    assert n > 0, f"no target edges for {task}"
PY

# ---- 2. training: one model per family, both tasks --------------------------
step "train: distmult on Task A" python train_and_eval.py --tsv "$TSV_A" --task "$TASK_A" \
  --model distmult --config "$(PROTOCOL=v2 resolve_config "$TASK_A")" --runs 1 --epochs "$EP" \
  $TRAIN_FLAGS --dry_run
step "train: rgcn on Task A (small graph)" python train_and_eval.py \
  --tsv dataset/PKT_subgraphs/ablation/pkt_ablA_core_ppi.tsv.zip --task "$TASK_A" \
  --model rgcn --config "$(PROTOCOL=v2 resolve_config "$TASK_A")" --runs 1 --epochs "$EP" \
  $TRAIN_FLAGS --dry_run
step "train: compgcn on Task A (small graph)" python train_and_eval.py \
  --tsv dataset/PKT_subgraphs/ablation/pkt_ablA_core_ppi.tsv.zip --task "$TASK_A" \
  --model compgcn --config "$(PROTOCOL=v2 resolve_config "$TASK_A")" --runs 1 --epochs "$EP" \
  $TRAIN_FLAGS --dry_run
step "train: distmult on Task B" python train_and_eval.py --tsv "$TSV_B" --task "$TASK_B" \
  --model distmult --config "$(PROTOCOL=v2 resolve_config "$TASK_B")" --runs 1 --epochs "$EP" \
  $TRAIN_FLAGS --dry_run

# ---- 3. ablation graphs: every variant trains -------------------------------
for f in dataset/PKT_subgraphs/ablation/pkt_ablA_*.tsv.zip; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in pkt_ablA_full.tsv.zip|pkt_ablA_core_ppi.tsv.zip) ;; *) continue ;; esac
  step "ablation: $(basename "$f" .tsv.zip)" python train_and_eval.py --tsv "$f" --task "$TASK_A" \
    --model rgcn --config "$(PROTOCOL=v2 resolve_config "$TASK_A")" --runs 1 --epochs 2 \
    $TRAIN_FLAGS --dry_run
done

# ---- 4. baselines -----------------------------------------------------------
step "baseline: popularity Task A" python experiments/baseline_popularity.py \
  --tsv "$TSV_A" --task "$TASK_A" --split_seed "$V2_SPLIT_SEED"
step "baseline: popularity Task B" python experiments/baseline_popularity.py \
  --tsv "$TSV_B" --task "$TASK_B" --split_seed "$V2_SPLIT_SEED"

# ---- 5. a real (non-dry) run, so that E4 has a model to load ----------------
step "train: saved model for E4" python train_and_eval.py --tsv "$TSV_A" --task "$TASK_A" \
  --model distmult --config "$(PROTOCOL=v2 resolve_config "$TASK_A")" --runs 1 --epochs 2 $TRAIN_FLAGS
MODEL_DIR="$(ls -td models/dti_pkt_taskA_dti* 2>/dev/null | head -1)"
if [ -n "$MODEL_DIR" ]; then
  step "E4: drug_eval on that model" python drug_eval.py --model_folder "$MODEL_DIR" \
    --tsv "$TSV_A" --task "$TASK_A" --target_type "$TGT_A" --topk 5 --candidate_pool relation
else
  echo "[smoke] E4: drug_eval                     SKIP (no model directory found)"; SKIP=$((SKIP+1))
fi

# ---- 6. summaries -----------------------------------------------------------
step "summary: protocol_compare_summary" python experiments/protocol_compare_summary.py \
  --logdir "$SMOKE_DIR" --out "$SMOKE_DIR"
step "summary: e1_summary" python experiments/e1_summary.py --logdir "$SMOKE_DIR" --out "$SMOKE_DIR"
step "summary: convergence_summary" python experiments/convergence_summary.py --logdir "$SMOKE_DIR"

# ---- 7. HPO (optional: needs wandb) ----------------------------------------
if [ "${SMOKE_HPO:-0}" = "1" ]; then
  step "HPO: one trial on Task A" env PKT_TSV="$TSV_A" PKT_TASK="$TASK_A" PKT_HPO_RUNS=1 \
    PKT_HPO_EPOCHS=2 PKT_HPO_PATIENCE=1 PKT_HPO_MODELS=distmult WANDB_PROJECT=RelationalPKT-smoke \
    python tuning_hyperparameter.py
else
  echo "[smoke] HPO: one trial                    SKIP (set SMOKE_HPO=1, needs wandb login)"; SKIP=$((SKIP+1))
fi

echo
echo "================ SMOKE TEST SUMMARY ================"
printf '%s\n' "${RESULTS[@]}"
echo "----------------------------------------------------"
echo "PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP   (logs in $SMOKE_DIR)"
[ "$FAIL" -eq 0 ] || exit 1
