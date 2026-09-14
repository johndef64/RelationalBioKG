#!/usr/bin/env bash
# E2 (tandem) — run BOTH hyperparameter sweeps, Task A then Task B, with a reduced budget.
# Sequential by design: one GPU can't train two sweeps at once (they'd OOM / slow each other).
# Each e2_hpo_sweep.sh sweeps the models of tuning_hyperparameter.py (v2: rgcn + compgcn + distmult baseline;
# v1: rgcn + compgcn; override with PKT_HPO_MODELS) x PKT_HPO_RUNS configs per task.
#
# Usage:   bash experiments/e2_hpo_tandem.sh                 # 30 runs/model (default)
#          PKT_HPO_RUNS=20 bash experiments/e2_hpo_tandem.sh # lighter
# Prereq:  wandb login   (entity/project hardcoded to RelationalPKT in tuning_hyperparameter.py)
set -euo pipefail
cd "$(dirname "$0")/.."

export PKT_HPO_RUNS="${PKT_HPO_RUNS:-30}"          # configs per model (v2: smaller space than v1)
# Early stopping is the main time lever. From the E1 log a good config converges by ~epoch
# 105 and would early-stop ~125; the tuning default (patience 50 evals) never triggers inside
# 200 epochs, so every trial wastes the full budget. patience=6 (=~35 epochs of no gain) stops
# plateaued trials ~epoch 125-140 -> ~35% faster, without cutting still-improving configs.
export PKT_HPO_PATIENCE="${PKT_HPO_PATIENCE:-6}"   # evaluations (every 5 epochs) w/o improvement
export PKT_HPO_EPOCHS="${PKT_HPO_EPOCHS:-200}"     # hard ceiling (rarely reached with the above)

if [ "${PKT_HPO_PROTOCOL:-v2}" = "v2" ]; then N_MODELS=$(echo ${PKT_HPO_MODELS:-rgcn compgcn distmult} | wc -w); else N_MODELS=$(echo ${PKT_HPO_MODELS:-rgcn compgcn} | wc -w); fi
echo "[E2-tandem] protocol=${PKT_HPO_PROTOCOL:-v2} PKT_HPO_RUNS=$PKT_HPO_RUNS per model x ${N_MODELS} models  ->  $((PKT_HPO_RUNS*N_MODELS)) runs/task, $((PKT_HPO_RUNS*N_MODELS*2)) total"
echo "[E2-tandem] === Task A (DTI) -> project RelationalPKT-DTI ==="
bash experiments/e2_hpo_sweep.sh A
echo "[E2-tandem] === Task B (TREATS) -> project RelationalPKT-TREATS ==="
bash experiments/e2_hpo_sweep.sh B

# Extract the best config per task straight from W&B (best-effort) and inject into
# src/models_params.json as PKT-DTI-best / PKT-TREATS-best.
echo "[E2-tandem] extracting best configs from W&B ..."
if [ "${PKT_HPO_PROTOCOL:-v2}" = "v2" ]; then SUFFIX="${HPO_SUFFIX:--v2}"; else SUFFIX="${HPO_SUFFIX:-}"; fi
python experiments/get_best_hpo_config.py --task DTI    --suffix "$SUFFIX" --write || echo "  (skip DTI: run get_best_hpo_config.py manually)"
python experiments/get_best_hpo_config.py --task TREATS --suffix "$SUFFIX" --write || echo "  (skip TREATS: run get_best_hpo_config.py manually)"
echo "[E2-tandem] done. Best configs saved (experiments/hpo_best/) and injected as PKT-<TASK>-best${SUFFIX}."
echo "[E2-tandem] Final training:  PROTOCOL=v2 bash experiments/e1_main_training.sh   (resolve_config picks PKT-<TASK>-best${SUFFIX})"
